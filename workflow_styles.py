def get_workflow_css():
    return """
    <style>
    .lane {
        border: 1px solid #ddd;
        border-radius: 10px;
        margin: 1.5em 0;
        background-color: #ffffff;
        font-family: sans-serif;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
    }

    .header {
        padding: 10px;
        font-weight: bold;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        color: white;
    }

    .model { background-color: #0d6efd; }
    .stress { background-color: #198754; }
    .rationale { background-color: #6f42c1; }

    .horizontal-body {
        padding: 1em;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }

    .step-model, .step-stress, .step-rationale {
        padding: 10px 14px;
        background: #f0f0f0;
        border-radius: 8px;
        font-size: 0.9em;
        font-weight: 500;
        color: #333;
        position: relative;
        opacity: 0;
        animation: fadeIn 0.6s ease-in forwards;
    }

    .step-model.completed,
    .step-stress.completed,
    .step-rationale.completed {
        background-color: #dff0d8;
        border: 1px solid #198754;
        color: #155724;
    }

    .step-model.completed::after,
    .step-stress.completed::after,
    .step-rationale.completed::after {
        content: "✔";
        position: absolute;
        right: -12px;
        top: -12px;
        background: #198754;
        color: white;
        border-radius: 50%;
        width: 18px;
        height: 18px;
        text-align: center;
        font-size: 0.75em;
        line-height: 18px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }

    .arrow-model, .arrow-stress, .arrow-rationale {
        font-size: 1em;
        line-height: 1;
        transform: scaleX(1.2);
    }

    .arrow-model { color: #0d6efd; }
    .arrow-stress { color: #198754; }
    .arrow-rationale { color: #6f42c1; }

    @keyframes fadeIn {
        to {
            opacity: 1;
        }
    }
    .pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 3px 10px;
    font-size: 9px;
    font-weight: 500;
    color: white;
    border-radius: 9999px;
    min-width: 110px;
    text-align: center;
    line-height: 1.4;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    }
    .pill.llm { background-color: #00A3C4; }
    .pill.func { background-color: #b19cd9; }
    .pill.success { background-color: #28a745; }
    .pill.warn { background-color: #f39c12; }


    </style>
    """


def get_workflow_html_ml(step):
    def box(text, index):
        class_name = "step-model completed" if step > index else "step-model"
        delay = f"{0.2 * index:.1f}s"
        return f'<div class="{class_name}" style="animation-delay: {delay};">{text}</div>'

    html = f"""
    <div class="lane">
        <div class="header model">Model Inference</div>
        <div class="horizontal-body">
            {box("Construct Model Input", 0)}
            <div class="arrow-model">▶</div>
            {box("Send to Azure Machine Learning Model", 1)}
            <div class="arrow-model">▶</div>
            {box("Model Execution", 2)}
            <div class="arrow-model">▶</div>
            {box("Fair Value Observability Prediction", 2)}
        </div>
    </div>
    """
    return html

def get_workflow_html_rf(step):
    def box(text, index):
        class_name = "step-stress completed" if step > index else "step-stress"
        delay = f"{0.2 * index:.1f}s"
        return f'<div class="{class_name}" style="animation-delay: {delay};">{text}</div>'

    html = f"""
    <div class="lane">
        <div class="header stress">Risk Factor Observability</div>
        <div class="horizontal-body">
            {box("Identify Risk Factors", 0)}
            <div class="arrow-stress">▶</div>
            {box("Test IR Delta Observability", 1)}
            <div class="arrow-stress">▶</div>
            {box("Test Volatality Observability", 2)}
            <div class="arrow-stress">▶</div>
            {box("Assess observability of Total PV", 3)}
        </div>
    </div>
    """
    return html

def get_workflow_html_rat(step):
    def box(text, index):
        class_name = "step-rationale completed" if step > index else "step-rationale"
        delay = f"{0.2 * index:.1f}s"
        return f'<div class="{class_name}" style="animation-delay: {delay};">{text}</div>'

    html = f"""
    <div class="lane">
        <div class="header rationale">Ground Model Prediction</div>
        <div class="horizontal-body">
            {box("Analytical Review", 0)}
            <div class="arrow-rationale">▶</div>
            {box("Generate Commentary", 1)}
            <div class="arrow-rationale">▶</div>
            {box("Summarize Insight", 2)}
        </div>
    </div>
    """
    return html
