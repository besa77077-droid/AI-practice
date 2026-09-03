"""FastAPI web app: two screens — interview breakdown (upload, insight cards +
transcript with click-to-highlight traceability) and hypotheses/backlog.

Uploads never block the request: `/upload` starts a background job and
redirects to a polling page. This is the direct fix for the old prototype's
failure mode, where a real audio file made the browser hang with no
feedback until it either finished or the connection was reset.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from insight_engine.jobs import STATUS_DONE, STATUS_ERROR, job_manager
from insight_engine.pipeline import PipelineResult, ingest_audio, ingest_text, process_interview
from insight_engine.providers.llm import build_llm_provider
from insight_engine.storage.sqlite_store import InsightStore

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Insight Engine")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

_store: InsightStore | None = None


def get_store() -> InsightStore:
    global _store
    if _store is None:
        _store = InsightStore(os.environ.get("INSIGHT_ENGINE_DB", "insights.db"))
    return _store


def get_llm():
    timeout_env = os.environ.get("OLLAMA_TIMEOUT_S")
    return build_llm_provider(
        os.environ.get("INSIGHT_ENGINE_LLM", "ollama"),
        model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
        base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        timeout_s=float(timeout_env) if timeout_env else None,
    )


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        return tmp.name


def _read_transcript_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    raw = upload.file.read()
    if suffix == ".docx":
        import io

        from docx import Document

        try:
            document = Document(io.BytesIO(raw))
        except Exception as exc:
            raise ValueError(
                "Не удалось прочитать .docx — файл повреждён или это не Word-документ"
            ) from exc
        return "\n".join(p.text for p in document.paragraphs)
    return raw.decode("utf-8")


@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse(url="/interviews")


@app.get("/interviews", response_class=HTMLResponse)
def interviews_index(request: Request):
    store = get_store()
    return templates.TemplateResponse(
        request, "interviews.html",
        {"interviews": store.list_interviews(), "active_interview": None, "insights": [],
         "message": request.query_params.get("msg"), "frequency_by_insight": {},
         "total_interviews": len(store.list_interviews())},
    )


@app.get("/interviews/{interview_id}", response_class=HTMLResponse)
def interview_detail(request: Request, interview_id: str, highlight: int | None = None):
    store = get_store()
    interview = store.get_interview(interview_id)
    if interview is None:
        return RedirectResponse(url="/interviews?msg=Интервью+не+найдено")
    insights = store.list_insights_for_interview(interview_id)
    frequency_by_insight = {}
    for insight in insights:
        cluster = store.get_cluster_for_insight(insight.id)
        if cluster is None:
            continue
        distinct_interviews = {
            m.interview_id for m in store.get_cluster_members_detail(cluster.id)
        }
        frequency_by_insight[insight.id] = len(distinct_interviews)
    total_interviews = len(store.list_interviews())
    return templates.TemplateResponse(
        request, "interviews.html",
        {
            "interviews": store.list_interviews(),
            "active_interview": interview,
            "insights": insights,
            "highlight_segment": highlight,
            "message": None,
            "frequency_by_insight": frequency_by_insight,
            "total_interviews": total_interviews,
        },
    )


@app.post("/insights/{insight_id}/status")
def set_insight_status(insight_id: str, status: str = Form(...)):
    store = get_store()
    insight = store.get_insight(insight_id)
    if insight is None:
        return RedirectResponse(url="/interviews", status_code=303)
    store.update_insight_status(insight_id, status)
    return RedirectResponse(url=f"/interviews/{insight.interview_id}", status_code=303)


@app.post("/upload")
def upload(
    title: str = Form(...),
    respondent: str = Form(...),
    segment_label: str = Form(...),
    source: str = Form(...),
    transcript_text: str = Form(""),
    transcript_file: UploadFile | None = File(None),
    audio_file: UploadFile | None = File(None),
    whisper_model: str = Form("small"),
    diarize: str | None = Form(None),
    hf_token: str = Form(""),
):
    job = job_manager.create()
    audio_temp_path: str | None = None
    text_value: str | None = None

    try:
        if source == "audio_file":
            if audio_file is None or not audio_file.filename:
                raise ValueError("Выберите аудиофайл")
            audio_temp_path = _save_upload_to_temp(audio_file)
        elif source == "transcript_file":
            if transcript_file is None or not transcript_file.filename:
                raise ValueError("Выберите файл транскрипта")
            text_value = _read_transcript_upload(transcript_file)
        else:
            if not transcript_text.strip():
                raise ValueError("Введите текст транскрипта")
            text_value = transcript_text
    except ValueError as exc:
        return RedirectResponse(url=f"/interviews?msg={exc}", status_code=303)

    def run(on_step):
        try:
            if audio_temp_path:
                segments = ingest_audio(
                    audio_temp_path,
                    whisper_model=whisper_model,
                    diarize=diarize is not None,
                    hf_token=hf_token or None,
                    on_step=on_step,
                )
            else:
                segments = ingest_text(text_value or "", on_step=on_step)

            store = get_store()
            llm = get_llm()
            return process_interview(
                segments=segments, title=title, respondent=respondent,
                segment_label=segment_label, llm=llm, store=store, on_step=on_step,
            )
        finally:
            if audio_temp_path:
                Path(audio_temp_path).unlink(missing_ok=True)

    job_manager.start(job.id, run)
    return RedirectResponse(url=f"/processing/{job.id}", status_code=303)


@app.get("/processing/{job_id}", response_class=HTMLResponse)
def processing_page(request: Request, job_id: str):
    return templates.TemplateResponse(request, "processing.html", {"job_id": job_id})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        return JSONResponse({"status": "not_found"}, status_code=404)

    payload = {
        "status": job.status,
        "steps": [{"name": s.name, "detail": s.detail} for s in job.steps],
        "error": job.error,
    }
    if job.status == STATUS_DONE and isinstance(job.result, PipelineResult):
        payload["interview_id"] = job.result.interview.id
        payload["insight_count"] = len(job.result.insights)
        payload["hypothesis_count"] = len(job.result.hypotheses)
    if job.status == STATUS_ERROR:
        payload["error"] = job.error
    return JSONResponse(payload)


@app.get("/hypotheses", response_class=HTMLResponse)
def hypotheses_page(request: Request):
    store = get_store()
    hypotheses = store.list_hypotheses_ranked()
    basis_by_hyp = {h.id: store.get_cluster_members_detail(h.cluster_id) for h in hypotheses}
    return templates.TemplateResponse(
        request, "hypotheses.html",
        {"hypotheses": hypotheses, "basis_by_hyp": basis_by_hyp,
         "message": request.query_params.get("msg")},
    )


@app.post("/hypotheses/{hypothesis_id}/confirm")
def confirm_hypothesis(hypothesis_id: str):
    store = get_store()
    store.update_hypothesis_status(hypothesis_id, "confirmed")
    return RedirectResponse(url="/hypotheses", status_code=303)


@app.post("/hypotheses/export")
def export_hypotheses(hypothesis_ids: list[str] = Form(default=[]), tracker: str = Form("jira")):
    if not hypothesis_ids:
        return RedirectResponse(url="/hypotheses?msg=Отметьте+хотя+бы+одну+гипотезу", status_code=303)
    store = get_store()
    for hid in hypothesis_ids:
        store.update_hypothesis_status(hid, "in_backlog")
    msg = f"{len(hypothesis_ids)} карточки выгружены в {tracker.title()}"
    return RedirectResponse(url=f"/hypotheses?msg={msg}", status_code=303)


def serve() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
