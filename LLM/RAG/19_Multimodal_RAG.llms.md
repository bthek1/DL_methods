# Multimodal RAG

Multimodal RAG extends retrieval-augmented generation beyond text to include images, audio, video, and structured data — enabling richer knowledge bases and more capable question-answering systems.

Author

Benedict Thekkel

## 1. Why Multimodal RAG

Real-world knowledge is not purely text:

| Domain          | Non-text content                                       |
|-----------------|--------------------------------------------------------|
| Medical         | X-rays, MRI scans, pathology slides, radiology reports |
| Engineering     | CAD diagrams, schematics, technical drawings           |
| E-commerce      | Product images, demo videos                            |
| Education       | Textbook figures, lecture slides, recorded lectures    |
| Legal / Finance | Tables, charts, scanned contracts                      |

Text-only RAG loses information locked in these modalities. Multimodal RAG indexes and retrieves across all of them.

------------------------------------------------------------------------

## 2. Architectures: Three Approaches

### Approach A: Caption + embed text

Convert non-text content to text descriptions, then use a standard text RAG pipeline.

``` python
# Generate caption from image
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def image_to_caption(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(out[0], skip_special_tokens=True)

# Then embed the caption as a text chunk
caption = image_to_caption("diagram.png")
embedding = text_embedder.encode(caption)
vector_store.upsert(id="img_001", vector=embedding, metadata={"type": "image", "path": "diagram.png"})
```

**Pros:** Simple, works with any text-only LLM\
**Cons:** Captions lose visual detail; retrieval is only as good as the caption quality

### Approach B: Native multimodal embeddings

Use a multimodal encoder (CLIP, ImageBind) that maps different modalities into a shared embedding space.

``` python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Embed an image
image = preprocess(Image.open("diagram.png")).unsqueeze(0)
with torch.no_grad():
    image_embedding = model.encode_image(image)  # 512-dim vector

# Embed a text query — same space!
text = tokenizer(["circuit diagram showing capacitor arrangement"])
with torch.no_grad():
    text_embedding = model.encode_text(text)  # 512-dim vector

# Text query can now retrieve images via cosine similarity
```

**Pros:** True cross-modal retrieval (text query → image result)\
**Cons:** CLIP-based models limited to image-text; AudioBind/ImageBind needed for other modalities

### Approach C: LLM processes multimodal context

Use a vision-language model (GPT-4o, Claude 3, LLaVA, Gemini) as the generator. Retrieve images directly and pass them alongside text to the LLM.

``` python
from openai import OpenAI
import base64

client = OpenAI()

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What does this circuit diagram show?"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encode_image('diagram.png')}"
            }}
        ]
    }]
)
```

------------------------------------------------------------------------

## 3. Audio and Video

### Audio / Speech

1.  **Transcribe** speech to text using Whisper or a cloud ASR service
2.  Chunk the transcript (by time segment or by sentence)
3.  Embed and index as text — preserve timestamp metadata for source attribution

``` python
import whisper

model = whisper.load_model("large")
result = model.transcribe("lecture.mp3", word_timestamps=True)

# Segment into ~30s chunks preserving timestamps
chunks = []
for segment in result["segments"]:
    chunks.append({
        "text": segment["text"],
        "start": segment["start"],
        "end": segment["end"],
        "source": "lecture.mp3",
    })
```

### Video

Video = frames + audio. Common approach: 1. Extract audio → transcribe → chunk transcripts 2. Extract keyframes (scene changes) → caption each frame 3. Store both with aligned timestamps 4. At retrieval time, return transcript chunk + nearest keyframe

``` python
import cv2

def extract_keyframes(video_path: str, interval_sec: int = 30) -> list:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % (int(fps) * interval_sec) == 0:
            frames.append((frame_num / fps, frame))
        frame_num += 1
    return frames
```

------------------------------------------------------------------------

## 4. Structured and Tabular Data

Tables and databases are a distinct “modality” — they require query-time reasoning, not just similarity search.

### Text-to-SQL for structured data

``` python
TEXT_TO_SQL_PROMPT = """
You have access to a database with the following schema:
{schema}

Write a SQL query to answer: {question}
Return only the SQL.
"""

def answer_from_db(question: str, db_schema: str, db_conn) -> str:
    sql = llm(TEXT_TO_SQL_PROMPT.format(schema=db_schema, question=question))
    # Sanitise SQL before execution!
    result = db_conn.execute(sql).fetchall()
    return llm(f"Summarise this query result: {result}\nQuestion: {question}")
```

**Security note:** Never execute LLM-generated SQL directly on a production database. Use a read-only connection, validate against an allowlist of tables, and log all queries.

### Routing: text vs structured retrieval

``` python
def route_query(question: str) -> str:
    """Return 'vector' or 'sql' based on query type."""
    prompt = f"""
    Does this question require querying a database table (numbers, statistics,
    specific records) or searching document text (concepts, explanations, policies)?
    Answer 'sql' or 'vector' only.
    Question: {question}
    """
    return llm(prompt).strip().lower()
```

------------------------------------------------------------------------

## 5. Index Design for Multiple Modalities

``` python
# Separate indices per modality, unified retrieval interface
class MultimodalRAG:
    def __init__(self):
        self.text_index  = VectorStore(dim=1536)   # OpenAI embeddings
        self.image_index = VectorStore(dim=512)    # CLIP embeddings
        self.audio_index = VectorStore(dim=1536)   # Whisper transcripts → text embeddings

    def retrieve(self, query: str, modalities=["text", "image", "audio"]) -> list:
        results = []
        query_text_emb  = text_embedder.encode(query)
        query_image_emb = clip_model.encode_text(query)  # text → image space

        if "text"  in modalities: results += self.text_index.search(query_text_emb,  top_k=5)
        if "image" in modalities: results += self.image_index.search(query_image_emb, top_k=3)
        if "audio" in modalities: results += self.audio_index.search(query_text_emb,  top_k=3)

        return rerank(query, results)  # unified reranking across modalities

    def generate(self, query: str, results: list) -> str:
        # Separate images from text for the multimodal prompt
        text_chunks = [r for r in results if r.type == "text"]
        image_paths = [r.path for r in results if r.type == "image"]
        return vision_llm.generate(query, text_chunks, image_paths)
```

**Cross-modal relevance scoring challenge:** A highly relevant image and a moderately relevant text chunk have scores on different scales. Use modality-specific score normalisation before merging:

``` python
# Normalise scores within each modality to [0, 1] before merging
text_scores  = softmax([r.score for r in text_results])
image_scores = softmax([r.score for r in image_results])
```

------------------------------------------------------------------------

## 6. Source Attribution for Non-text Modalities

Users need to verify where answers come from — especially for images and audio.

``` python
# Return inline citations that link to the original asset
def format_answer_with_sources(answer: str, sources: list) -> dict:
    citations = []
    for s in sources:
        if s.type == "text":
            citations.append({"type": "text", "doc": s.source, "page": s.page})
        elif s.type == "image":
            citations.append({"type": "image", "path": s.path, "caption": s.caption})
        elif s.type == "audio":
            citations.append({"type": "audio", "file": s.source,
                               "timestamp": f"{s.start:.0f}s–{s.end:.0f}s"})
    return {"answer": answer, "sources": citations}
```

------------------------------------------------------------------------

## 7. Trade-offs and Practical Guidance

| Modality | Index cost | Retrieval quality | LLM support |
|----|----|----|----|
| Text | Low | Excellent | Universal |
| Images (captioned) | Medium | Good | Any text LLM |
| Images (CLIP) | Medium | Good (cross-modal) | Needs vision LLM for generation |
| Audio (transcribed) | Medium | Good (text) | Any text LLM |
| Video | High | Moderate | Needs vision LLM |
| Tables / SQL | Low | Excellent for structured queries | Any LLM with Text-to-SQL |

**Start with:** caption-based image indexing. It is simple, works with any existing text RAG pipeline, and covers 80% of use cases.

**Upgrade to** native multimodal embeddings (CLIP/ImageBind) when: - Users search for images without knowing how to describe them in text - Your domain has rich visual content where captions lose important detail

**Upgrade to** vision-language generation (GPT-4o, LLaVA) when: - The answer requires reasoning *about* the visual content, not just retrieving it - You have sufficient budget for the significantly higher inference cost

------------------------------------------------------------------------

Back to top
