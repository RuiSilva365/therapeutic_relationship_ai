from flask import Flask, render_template
import os
import json

app = Flask(__name__)

def load_latest_report():
    # Procura o arquivo de relatório mais recente na pasta "reports/"
    files = [f for f in os.listdir("reports") if f.startswith("report_") and f.endswith(".json")]
    if not files:
        return "Nenhum relatório encontrado."
    files.sort(reverse=True)
    with open(os.path.join("reports", files[0]), "r", encoding="utf-8") as f:
        report = json.load(f)
    return report

@app.route("/")
def home():
    report = load_latest_report()
    return render_template("index.html", report=report)

if __name__ == "__main__":
    app.run(debug=True)
