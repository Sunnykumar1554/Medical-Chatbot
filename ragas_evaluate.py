"""
ragas_evaluate.py
─────────────────────────────────────────────────────────────────────────────
RAGAS (Retrieval-Augmented Generation Assessment) evaluation for the
Medical Chatbot project.

RAGAS Metrics Used:
  • Faithfulness       – Is the answer grounded in the retrieved context?
  • Answer Relevancy   – Does the answer actually address the question?
  • Context Precision  – Are the retrieved chunks ranked well (relevant first)?
  • Context Recall     – Did retrieval capture all info needed for the answer?

Usage:
    python ragas_evaluate.py                        # full eval (OpenAI)
    python ragas_evaluate.py --samples 3            # quick smoke-test
    python ragas_evaluate.py --output results.json  # custom output file
"""

import os
import json
import time
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Validate required env vars ────────────────────────────────────────────────
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY is not set in your .env file.")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is required for RAGAS evaluation (it uses GPT to judge answers). "
        "Add it to your .env file: OPENAI_API_KEY=sk-..."
    )

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]   = OPENAI_API_KEY

# ── Imports ───────────────────────────────────────────────────────────────────
print("⏳ Loading dependencies...")

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.metrics.collections.context_precision import ContextPrecision
from ragas.metrics.collections.context_recall import ContextRecall

# ── Medical Test Dataset ──────────────────────────────────────────────────────
# ground_truth = the ideal / expected answer (used for Context Recall scoring)
TEST_DATASET = [
    {
        "question": "What are the common symptoms of diabetes?",
        "ground_truth": (
            "Common symptoms of diabetes include frequent urination, excessive thirst, "
            "unexplained weight loss, fatigue, blurred vision, slow healing wounds, "
            "and frequent infections. Type 1 may also cause nausea and vomiting."
        ),
    },
    {
        "question": "What causes high blood pressure (hypertension)?",
        "ground_truth": (
            "High blood pressure is caused by factors including obesity, high sodium diet, "
            "physical inactivity, stress, smoking, excessive alcohol, family history, "
            "age, and underlying conditions such as kidney disease or sleep apnea."
        ),
    },
    {
        "question": "What is the treatment for a migraine headache?",
        "ground_truth": (
            "Migraine treatment includes over-the-counter pain relievers (ibuprofen, "
            "acetaminophen), triptans for acute attacks, resting in a dark quiet room, "
            "cold compresses, and preventive medications like beta-blockers or topiramate "
            "for frequent migraines."
        ),
    },
    {
        "question": "What are the early signs of a heart attack?",
        "ground_truth": (
            "Early signs of a heart attack include chest pain or pressure, pain radiating "
            "to the arm, neck, or jaw, shortness of breath, nausea, cold sweats, "
            "dizziness, and fatigue. Women may experience atypical symptoms such as "
            "back pain or stomach discomfort."
        ),
    },
    {
        "question": "How is pneumonia diagnosed and treated?",
        "ground_truth": (
            "Pneumonia is diagnosed via chest X-ray, blood tests, and sputum culture. "
            "Treatment depends on the cause: bacterial pneumonia uses antibiotics, viral "
            "pneumonia may use antivirals, and both require rest, fluids, and fever "
            "reducers. Severe cases may need hospitalization and oxygen therapy."
        ),
    },
    {
        "question": "What are the symptoms and causes of anemia?",
        "ground_truth": (
            "Anemia symptoms include fatigue, weakness, pale skin, shortness of breath, "
            "dizziness, and cold hands. Causes include iron deficiency, vitamin B12 or "
            "folate deficiency, chronic disease, blood loss, or inherited conditions "
            "like sickle cell disease or thalassemia."
        ),
    },
    {
        "question": "What is the difference between Type 1 and Type 2 diabetes?",
        "ground_truth": (
            "Type 1 diabetes is an autoimmune condition where the body produces no "
            "insulin and requires daily insulin injections. Type 2 diabetes is a "
            "metabolic condition where the body doesn't use insulin effectively; it "
            "is managed through lifestyle changes, oral medications, and sometimes insulin."
        ),
    },
    {
        "question": "What are the side effects of ibuprofen?",
        "ground_truth": (
            "Common side effects of ibuprofen include stomach pain, nausea, heartburn, "
            "and dizziness. Serious side effects can include GI bleeding, kidney damage, "
            "increased blood pressure, and increased risk of heart attack or stroke, "
            "especially with long-term use."
        ),
    },
    {
        "question": "How can anxiety be managed without medication?",
        "ground_truth": (
            "Anxiety can be managed without medication through cognitive behavioral therapy "
            "(CBT), regular exercise, mindfulness and meditation, deep breathing techniques, "
            "adequate sleep, reducing caffeine and alcohol, and maintaining a strong "
            "social support network."
        ),
    },
    {
        "question": "What are the warning signs of a stroke?",
        "ground_truth": (
            "Stroke warning signs follow the FAST acronym: Face drooping, Arm weakness, "
            "Speech difficulty, Time to call emergency services. Other signs include "
            "sudden severe headache, vision problems, confusion, and loss of balance. "
            "Immediate emergency care is critical."
        ),
    },
]


# ── Helper: Run the RAG pipeline ──────────────────────────────────────────────
def build_rag_chain(use_openai_llm: bool = False):
    """Build the retrieval chain (mirrors app.py setup)."""
    embeddings = download_hugging_face_embeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings,
    )
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    if use_openai_llm:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    else:
        llm = ChatOllama(
            model=os.environ.get("LLAMA_MODEL", "llama3"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            temperature=float(os.environ.get("LLAMA_TEMPERATURE", "0.1")),
        )

    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt),
    )
    return chain, retriever


def run_rag(chain, question: str) -> dict:
    """Invoke RAG and return answer + context strings."""
    response   = chain.invoke({"input": question})
    answer     = response["answer"]
    contexts   = [doc.page_content for doc in response.get("context", [])]
    return {"answer": answer, "contexts": contexts}


async def run_all_questions(chain, dataset_slice):
    """Run all questions in parallel using a thread pool (3 workers)."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=3) as executor:
        tasks = [
            loop.run_in_executor(executor, run_rag, chain, item["question"])
            for item in dataset_slice
        ]
        return await asyncio.gather(*tasks)


# ── Main Evaluation ───────────────────────────────────────────────────────────
def main(n_samples: int, output_file: str, use_openai_llm: bool):

    print("\n" + "=" * 65)
    print("       🏥  MEDICAL CHATBOT — RAGAS EVALUATION")
    print("=" * 65)
    print(f"  Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Samples     : {n_samples}")
    print(f"  Chatbot LLM : {'OpenAI GPT-3.5' if use_openai_llm else 'Ollama/Llama3'}")
    print(f"  Judge LLM   : OpenAI GPT-3.5 (RAGAS)")
    print("=" * 65)

    # ── 1. Build RAG chain ────────────────────────────────────────────────────
    print("\n[1/4] Building RAG chain...")
    chain, _ = build_rag_chain(use_openai_llm)

    # ── 2. Collect answers & contexts ─────────────────────────────────────────
    dataset_slice = TEST_DATASET[:n_samples]
    samples       = []
    raw_results   = []

    print(f"\n[2/4] Running {n_samples} questions through the chatbot (parallel, 3 workers)...")
    t0_all = time.time()
    results = asyncio.run(run_all_questions(chain, dataset_slice))
    total_elapsed = round(time.time() - t0_all, 2)
    print(f"       ✓  All {n_samples} questions answered in {total_elapsed}s (parallel)")

    for i, (item, result) in enumerate(zip(dataset_slice, results), 1):
        q = item["question"]
        raw_results.append({
            "question"    : q,
            "answer"      : result["answer"],
            "contexts"    : result["contexts"],
            "ground_truth": item["ground_truth"],
            "latency_sec" : round(total_elapsed / n_samples, 2),
        })

        # Build RAGAS sample
        samples.append(
            SingleTurnSample(
                user_input         = q,
                response           = result["answer"],
                retrieved_contexts = result["contexts"],
                reference          = item["ground_truth"],
            )
        )
        print(f"  [{i}/{n_samples}] {q[:60]}... | {len(result['contexts'])} chunks")

    eval_dataset = EvaluationDataset(samples=samples)

    # ── 3. Configure RAGAS judge LLM & embeddings ─────────────────────────────
    print("\n[3/4] Configuring RAGAS judge (OpenAI GPT-3.5)...")

    judge_llm   = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", temperature=0))
    judge_embed = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embed),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    # ── 4. Run RAGAS evaluation ───────────────────────────────────────────────
    print("\n[4/4] Running RAGAS evaluation (this may take 1-3 minutes)...")
    ragas_result = evaluate(
        dataset      = eval_dataset,
        metrics      = metrics,
        raise_exceptions = False,
        show_progress    = True,
    )

    scores_df = ragas_result.to_pandas()

    # ── 5. Print Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("                    📊  RAGAS SCORES")
    print("=" * 65)

    metric_names = {
        "faithfulness"      : "Faithfulness      (answer grounded in context?)",
        "answer_relevancy"  : "Answer Relevancy  (answers the question?)",
        "context_precision" : "Context Precision (relevant chunks ranked first?)",
        "context_recall"    : "Context Recall    (all needed info retrieved?)",
    }

    aggregate_scores = {}
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in scores_df.columns:
            val = scores_df[col].dropna().mean()
            aggregate_scores[col] = round(float(val), 4)
            grade = "🟢" if val >= 0.75 else "🟡" if val >= 0.5 else "🔴"
            label = metric_names.get(col, col)
            print(f"  {grade}  {label:<50} {val:.4f}")

    overall = sum(aggregate_scores.values()) / len(aggregate_scores) if aggregate_scores else 0
    print("-" * 65)
    print(f"  {'OVERALL SCORE':<54} {overall:.4f}")
    print("=" * 65)

    # Grading
    if overall >= 0.75:
        verdict = "🟢 GOOD  — Your chatbot is performing well!"
    elif overall >= 0.5:
        verdict = "🟡 FAIR  — Some areas need improvement."
    else:
        verdict = "🔴 POOR  — Significant improvements recommended."
    print(f"\n  Verdict: {verdict}")

    # Per-question breakdown
    print("\n" + "-" * 65)
    print("  PER-QUESTION BREAKDOWN")
    print("-" * 65)
    for i, (rr, row) in enumerate(zip(raw_results, scores_df.itertuples()), 1):
        f  = getattr(row, "faithfulness",       None)
        ar = getattr(row, "answer_relevancy",   None)
        cp = getattr(row, "context_precision",  None)
        cr = getattr(row, "context_recall",     None)
        print(f"\n  Q{i}: {rr['question'][:60]}...")
        print(f"       Latency          : {rr['latency_sec']}s")
        print(f"       Faithfulness     : {f:.3f}" if f is not None else "       Faithfulness     : N/A")
        print(f"       Answer Relevancy : {ar:.3f}" if ar is not None else "       Answer Relevancy : N/A")
        print(f"       Context Precision: {cp:.3f}" if cp is not None else "       Context Precision: N/A")
        print(f"       Context Recall   : {cr:.3f}" if cr is not None else "       Context Recall   : N/A")
        print(f"       Answer preview   : {rr['answer'][:120]}...")

    # ── 6. Improvement Suggestions ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  💡  IMPROVEMENT SUGGESTIONS")
    print("=" * 65)

    if aggregate_scores.get("faithfulness", 1) < 0.6:
        print("""
  ⚠️  LOW FAITHFULNESS — Bot is hallucinating (adding info not in context)
     Fix: Tighten your system_prompt:
          "Answer ONLY using the provided context. Do not add external info."
     Fix: Reduce LLM temperature (set LLAMA_TEMPERATURE=0.0 in .env)""")

    if aggregate_scores.get("answer_relevancy", 1) < 0.6:
        print("""
  ⚠️  LOW ANSWER RELEVANCY — Answers are off-topic
     Fix: Add more medical Q&A data to Pinecone (increase MAX_DOCS in store_csv_index.py)
     Fix: Refine the system prompt to stay focused on the patient's question""")

    if aggregate_scores.get("context_precision", 1) < 0.6:
        print("""
  ⚠️  LOW CONTEXT PRECISION — Irrelevant chunks retrieved first
     Fix: Switch retriever to MMR in app.py:
          retriever = docsearch.as_retriever(
              search_type="mmr",
              search_kwargs={"k": 5, "fetch_k": 20}
          )
     Fix: Improve chunk size (try chunk_size=800 in src/helper.py)""")

    if aggregate_scores.get("context_recall", 1) < 0.6:
        print("""
  ⚠️  LOW CONTEXT RECALL — Missing relevant info during retrieval
     Fix: Increase k (retrieved chunks) in app.py:
          search_kwargs={"k": 5}   # was 3
     Fix: Add more PDF medical textbooks to your data/ folder
     Fix: Increase chunk overlap (chunk_overlap=100 in src/helper.py)""")

    # ── 7. Save full JSON report ──────────────────────────────────────────────
    report = {
        "timestamp"        : datetime.now().isoformat(),
        "n_samples"        : n_samples,
        "chatbot_llm"      : "openai-gpt3.5" if use_openai_llm else "ollama-llama3",
        "aggregate_scores" : aggregate_scores,
        "overall_score"    : round(overall, 4),
        "per_question"     : [
            {
                **rr,
                "faithfulness"      : getattr(row, "faithfulness",       None),
                "answer_relevancy"  : getattr(row, "answer_relevancy",   None),
                "context_precision" : getattr(row, "context_precision",  None),
                "context_recall"    : getattr(row, "context_recall",     None),
            }
            for rr, row in zip(raw_results, scores_df.itertuples())
        ],
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅  Full report saved → {output_file}")
    print("=" * 65 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for Medical Chatbot")
    parser.add_argument(
        "--samples", type=int, default=len(TEST_DATASET),
        help=f"Number of test questions to evaluate (max {len(TEST_DATASET)})",
    )
    parser.add_argument(
        "--output", type=str, default="ragas_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--openai-llm", action="store_true",
        help="Use OpenAI GPT-3.5 as the chatbot LLM (instead of Ollama)",
    )
    args = parser.parse_args()

    n = min(args.samples, len(TEST_DATASET))
    main(n_samples=n, output_file=args.output, use_openai_llm=args.openai_llm)