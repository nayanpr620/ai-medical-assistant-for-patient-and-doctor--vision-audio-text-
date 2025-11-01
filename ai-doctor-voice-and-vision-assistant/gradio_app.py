# gradio_app.py
# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

import os
import json
import gradio as gr
from pathlib import Path
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

# --- Configuration ---
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"  # your choice A
GROQ_KEY = os.environ.get("GROQ_API_KEY")

# System prompt tuned for Format 2 + JSON output (analysis short, treatment longer)
SYSTEM_PROMPT_TEMPLATE = """
You are a professional medical doctor speaking directly to a patient. 
You will be given an image to analyze and an optional transcription of the patient's spoken context.
Produce a JSON object (and nothing else) with exactly two keys: "analysis" and "treatment".

Rules:
- "analysis": one or two concise sentences describing what appears medically wrong in the image. Use a natural doctor voice, start with "With what I see, ..." and keep it short.
- "treatment": 3 to 4 sentences giving practical remedies, next steps, and when to seek professional care. Use a natural doctor voice, clear steps, no bullet points or numbered lists.
- Do not include any extra keys, commentary, or markdown. Respond ONLY with valid JSON.

If the image is missing or unclear, set "analysis" to "Image not provided or unclear" and give a general short treatment in "treatment".
"""

# --- Helpers ---
def safe_parse_json(model_text: str):
    """
    Try to parse model_text as JSON. If it fails, do a best-effort extraction:
    look for the first {...} block and parse that. If still fails, return
    fallback values.
    """
    # direct parse
    try:
        return json.loads(model_text)
    except Exception:
        pass

    # try to find first JSON object in text
    start = model_text.find("{")
    end = model_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            candidate = model_text[start:end+1]
            return json.loads(candidate)
        except Exception:
            pass

    # fallback: return the whole text in treatment and a generic analysis
    return {
        "analysis": "Could not parse structured analysis from the model output.",
        "treatment": model_text.strip()[:2000]  # keep the raw text as treatment fallback
    }

# --- Core processing function ---
def process_inputs(audio_filepath, image_filepath):
    # 1) STT (optional)
    stt_text = ""
    try:
        if audio_filepath:
            stt_text = transcribe_with_groq(GROQ_API_KEY=GROQ_KEY, audio_filepath=audio_filepath, stt_model="whisper-large-v3") or ""
        else:
            stt_text = ""
    except Exception as e:
        stt_text = f"[STT error: {str(e)}]"

    # 2) Build prompt for LLM
    assembled_prompt = SYSTEM_PROMPT_TEMPLATE + "\n\n"
    if stt_text:
        assembled_prompt += "Patient speech (transcription): " + stt_text + "\n\n"

    if image_filepath:
        # encode image and pass it to your analyze function which should send image + prompt to the model
        try:
            encoded = encode_image(image_filepath)
            model_raw_output = analyze_image_with_query(query=assembled_prompt, encoded_image=encoded, model=MODEL_NAME)
        except Exception as e:
            model_raw_output = json.dumps({
                "analysis": "Image processing error",
                "treatment": f"Failed to analyze image due to error: {str(e)}"
            })
    else:
        # If no image, instruct the model accordingly
        assembled_prompt += "No image provided."
        try:
            model_raw_output = analyze_image_with_query(query=assembled_prompt, encoded_image=None, model=MODEL_NAME)
        except Exception as e:
            model_raw_output = json.dumps({
                "analysis": "Image not provided or unclear",
                "treatment": f"No image was provided. If you can, please upload a clear photo. Error detail: {str(e)}"
            })

    # 3) Parse model output (expect JSON)
    parsed = safe_parse_json(model_raw_output)

    # ensure keys exist
    analysis = parsed.get("analysis", "Analysis not available.")
    treatment = parsed.get("treatment", "Treatment not available.")

    # 4) Generate TTS audio for the doctor's combined response (you can choose what text to read out)
    # we'll speak a short combined voice output: analysis + treatment summary
    tts_text = f"{analysis} {treatment}"
    tts_path = "final.mp3"
    try:
        # ensure previous file removed to avoid conflicts
        if Path(tts_path).exists():
            Path(tts_path).unlink()
        text_to_speech_with_gtts(input_text=tts_text, output_filepath=tts_path)
    except Exception as e:
        # if TTS fails, keep audio empty and append error to treatment
        tts_path = None
        treatment = f"{treatment}\n\n[TTS generation failed: {str(e)}]"

    # 5) Return values in the order: Speech-to-text, analysis, treatment, audio filepath (or None)
    return stt_text, analysis, treatment, tts_path

# --- Build Enhanced UI with gradio Blocks ---
css = """
/* small medical-style theme */
body { font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue'; }
.gradio-container { --primary-hue: 207; } /* blue tone */
.header { text-align:center; margin-bottom: 12px; }
.app-card { background: #ffffff; border-radius: 12px; padding: 18px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
.label-quiet { color: #6b7280; font-size: 13px; }
.big-title { font-size: 26px; font-weight:700; color:#0b3b66; }
.small-note { color:#475569; font-size:13px; }
"""

with gr.Blocks(css=css, title="AI response with Vision and Voice (Medical)") as demo:
    with gr.Column(elem_id="top", scale=1):
        gr.Markdown("<div class='header'><div class='big-title'>AI response with Vision and Voice</div>"
                    "<div class='small-note'>Record or upload voice, upload a medical image, then press Submit</div></div>")

    with gr.Row():
        # Left column: inputs
        with gr.Column(scale=6):
          with gr.Group(elem_id="left_group", visible=True):
                gr.Markdown("#### Patient Input", elem_classes="label-quiet")
                audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Record Voice (click to start/stop)")
                image_in = gr.Image(type="filepath", label="Upload Medical Image (optional)")
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")

        # Right column: outputs
        with gr.Column(scale=6):
            with gr.Group(elem_id="right_card", visible=True):
                gr.Markdown("#### Results", elem_classes="label-quiet")
                stt_out = gr.Textbox(label="Speech to Text", interactive=False)
                analysis_out = gr.Textbox(label="Medical Analysis (1-2 sentences)", interactive=False)
                treatment_out = gr.Textbox(label="Treatment / Next Steps (3-4 sentences)", interactive=False)
                audio_out = gr.Audio(label="response output(playable)", interactive=False)

                # small note and flag
                flag_btn = gr.Button("Flag", visible=True)
                status = gr.Label(value="", visible=False)

    # Action wiring
    def on_submit(audio, image):
        # UI feedback while processing
        status_msg = "Processing... this may take a few seconds"
        return gr.update(value=status_msg), *process_inputs(audio, image)

    # When user clicks Submit -> call process_inputs and update outputs
    submit_btn.click(fn=process_inputs, inputs=[audio_in, image_in], outputs=[stt_out, analysis_out, treatment_out, audio_out])

    # Clear button resets inputs & outputs
    def clear_all():
        return None, None, None, None, None
    clear_btn.click(lambda: (None, None, "", "", "", None), inputs=None, outputs=[audio_in, image_in, stt_out, analysis_out, treatment_out, audio_out])

    # Optional: flag button to capture attention (no-op here, replace with logging)
    def flag_action(analysis_text, treatment_text):
        # you could log flagged cases to a database here
        return gr.update(value="Flagged — thanks. We'll review this case."), 
    flag_btn.click(flag_action, inputs=[analysis_out, treatment_out], outputs=[status])

    # Launch
    demo.launch(debug=True, share=False)