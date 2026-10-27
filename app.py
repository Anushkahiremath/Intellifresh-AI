from __future__ import annotations
import os, uuid
from pathlib import Path
from flask import Flask, abort, flash, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
from services.database import close_db, dashboard_stats, delete_analysis, get_analysis, init_db, list_analyses, save_analysis
from services.gradcam import create_gradcam
from services.predictor import FreshnessPredictor, ProducePredictor, ModelUnavailableError
from services.recommendations import DISCLAIMER, get_recommendation

BASE_DIR=Path(__file__).resolve().parent
UPLOAD_DIR=BASE_DIR/"static"/"uploads"; RESULT_DIR=BASE_DIR/"static"/"results"
UPLOAD_DIR.mkdir(parents=True,exist_ok=True); RESULT_DIR.mkdir(parents=True,exist_ok=True)
ALLOWED_EXTENSIONS={"png","jpg","jpeg","webp"}

app=Flask(__name__); app.config.update(SECRET_KEY=os.getenv("SECRET_KEY","dev-change-me"),MAX_CONTENT_LENGTH=10*1024*1024,DATABASE=str(BASE_DIR/"instance"/"freshvision.db"),MODEL_PATH=str(BASE_DIR/"models"/"freshness_mobilenetv2_state_dict.pth"))
predictor=FreshnessPredictor(app.config["MODEL_PATH"])
produce_predictor=ProducePredictor(BASE_DIR/"models"/"produce_efficientnet_b0.pth",BASE_DIR/"models"/"produce_config.json")
app.teardown_appcontext(close_db)
with app.app_context(): init_db()

def save_upload(upload):
    if not upload or not upload.filename: raise ValueError("Please select an image.")
    safe=secure_filename(upload.filename); ext=safe.rsplit(".",1)[-1].lower() if "." in safe else ""
    if ext not in ALLOWED_EXTENSIONS: raise ValueError("Supported formats are JPG, JPEG, PNG and WebP.")
    filename=f"{uuid.uuid4().hex}.{ext}"; path=UPLOAD_DIR/filename; upload.save(path)
    try:
        with Image.open(path) as image: image.verify()
    except (UnidentifiedImageError,OSError): path.unlink(missing_ok=True); raise ValueError("The uploaded file is not a valid image.")
    return filename,path

def analyze_upload(upload):
    filename,path=save_upload(upload)
    try: result=predictor.predict(path)
    except Exception: path.unlink(missing_ok=True); raise
    analysis_uuid=uuid.uuid4().hex; cam_name=f"{analysis_uuid}_gradcam.jpg"
    if not create_gradcam(predictor.model,result.pop("tensor"),result.pop("class_index"),path,RESULT_DIR/cam_name): cam_name=None
    record={**result,**produce_predictor.predict(path),"analysis_uuid":analysis_uuid,"image_filename":filename,"gradcam_filename":cam_name}
    save_analysis(record); return record

@app.get("/")
def index(): return render_template("index.html",model=predictor,produce_model=produce_predictor)
@app.route("/analyze",methods=["GET","POST"])
def analyze():
    if request.method=="POST":
        try: return redirect(url_for("result",analysis_id=analyze_upload(request.files.get("image"))["analysis_uuid"]))
        except (ValueError,ModelUnavailableError,RuntimeError) as exc: flash(str(exc),"danger")
        except Exception: app.logger.exception("Prediction failed"); flash("Prediction could not be completed. Please try another image.","danger")
    return render_template("analyze.html",model=predictor)
@app.get("/result/<analysis_id>")
def result(analysis_id):
    row=get_analysis(analysis_id)
    if not row: abort(404)
    return render_template("result.html",analysis=row,recommendation=get_recommendation(row["prediction"]),disclaimer=DISCLAIMER)
@app.route("/batch",methods=["GET","POST"])
def batch():
    if request.method=="POST":
        uploads=[item for item in request.files.getlist("images") if item.filename]
        if not uploads: flash("Select one or more images.","danger")
        elif len(uploads)>10: flash("A batch can contain at most 10 images.","danger")
        else:
            results=[]; errors=[]
            for upload in uploads:
                try: results.append(analyze_upload(upload))
                except Exception as exc: errors.append(f"{secure_filename(upload.filename)}: {exc}")
            return render_template("batch_results.html",results=results,errors=errors)
    return render_template("batch.html",model=predictor)
@app.get("/history")
def history(): return render_template("history.html",analyses=list_analyses())
@app.post("/history/<analysis_id>/delete")
def remove_history(analysis_id):
    row=delete_analysis(analysis_id)
    if row:
        (UPLOAD_DIR/row["image_filename"]).unlink(missing_ok=True)
        if row["gradcam_filename"]: (RESULT_DIR/row["gradcam_filename"]).unlink(missing_ok=True)
        flash("Analysis deleted.","success")
    return redirect(url_for("history"))
@app.get("/dashboard")
def dashboard():
    stats=dashboard_stats(); total=stats["total"] or 0
    fresh=stats["fresh"] or 0; stale=stats["stale"] or 0
    return render_template("dashboard.html",stats=stats,fresh_pct=round(fresh/total*100,1) if total else 0,stale_pct=round(stale/total*100,1) if total else 0)
@app.get("/model-info")
def model_info(): return render_template("model_info.html",model=predictor,produce_model=produce_predictor)
@app.get("/about")
def about(): return render_template("about.html")
@app.get("/health")
def health(): return {"status":"healthy","freshness_model_loaded":predictor.available,"produce_model_loaded":produce_predictor.available,"freshness_device":str(predictor.device),"produce_device":str(produce_predictor.device),"freshness_model_error":predictor.load_error,"produce_model_error":produce_predictor.load_error}
@app.errorhandler(413)
def too_large(_e): return render_template("error.html",code=413,message="Image is larger than the 10 MB limit."),413
@app.errorhandler(404)
def not_found(_e): return render_template("error.html",code=404,message="The requested page was not found."),404
@app.errorhandler(500)
def server_error(_e): return render_template("error.html",code=500,message="An unexpected error occurred."),500

if __name__=="__main__": app.run(host="0.0.0.0",port=int(os.getenv("PORT","5000")),debug=os.getenv("FLASK_DEBUG")=="1")
