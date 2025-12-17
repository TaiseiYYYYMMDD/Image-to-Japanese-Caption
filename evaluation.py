from caption_model import evaluate_caption_model

samples = [
    {
        "image_path": "static/uploads/dummy.jpg",
        "references": ["a man drinking water",
    "a person drinking from a bottle"]
    }
]

results = evaluate_caption_model(samples)
print(results)
