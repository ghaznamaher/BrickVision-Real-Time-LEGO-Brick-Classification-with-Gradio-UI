import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Loads the trained YOLO model
model = YOLO("best.pt")  # make sure best.pt is in the repo

def predict(image):
    results = model(image)  # run prediction
    annotated = results[0].plot()  # get annotated image
    return Image.fromarray(annotated)

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="LEGO Brick Vision",
    description="Upload a LEGO brick image to detect/classify."
)

if __name__ == "__main__":
    demo.launch()
