# app_interface.py
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from capstone_core import generate_report

# === Helper: compute analysis ===
def compute_analysis(df, threshold=0.3):
    df = df.copy()
    df["label"] = df["label"].astype(str).str.capitalize()
    df = df[df["label"].isin(["Polluted", "Unpolluted"])].reset_index(drop=True)

    species_cols = [c for c in df.columns if c.startswith("P_")][:8]
    env_cols = [c for c in ["pH", "moisture", "organic_content", "temperature", "nitrate"] if c in df.columns]

    X = df[species_cols + env_cols]
    y = df["label"].map({"Polluted": 1, "Unpolluted": 0})

    mask = X.dropna().index.intersection(y.dropna().index)
    X, y, df = X.loc[mask], y.loc[mask], df.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X_train_s, y_train)

    y_pred = rf.predict(X_test_s)
    y_proba = rf.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # --- Feature Importance ---
    fi_top = fi.head(15).sort_values()
    fig_imp = go.Figure(go.Bar(
        x=fi_top.values,
        y=fi_top.index,
        orientation='h',
        marker=dict(color=fi_top.values, colorscale='Viridis')
    ))
    fig_imp.update_layout(title="Feature Importance", template="plotly_dark", height=450)

    # --- Environmental Features ---
    if env_cols:
        env_melt = df.melt(id_vars="label", value_vars=env_cols, var_name="Feature", value_name="Value")
        fig_env = px.box(env_melt, x="Feature", y="Value", color="label", template="plotly_dark")
        fig_env.update_layout(title="Environmental Feature Distribution")
    else:
        fig_env = None

    # --- Species Differences ---
    if species_cols:
        species_means = df.groupby("label")[species_cols].mean().T
        species_means["diff"] = (species_means["Polluted"] - species_means["Unpolluted"]).abs()
        top = species_means.sort_values("diff", ascending=False).head(10).reset_index().rename(columns={"index": "species"})
        fig_species = go.Figure()
        fig_species.add_bar(x=top["species"], y=top["Polluted"], name="Polluted")
        fig_species.add_bar(x=top["species"], y=top["Unpolluted"], name="Unpolluted")
        fig_species.update_layout(barmode="group", template="plotly_dark", title="Top Species Differences")
    else:
        fig_species = None

    # --- Statewise Distribution ---
    state_col = next((c for c in ["state", "State", "location", "Location"] if c in df.columns), None)
    if state_col:
        state_counts = df.groupby([state_col, "label"]).size().unstack(fill_value=0).reset_index()
        fig_state = go.Figure()
        if "Polluted" in state_counts:
            fig_state.add_bar(x=state_counts[state_col], y=state_counts["Polluted"], name="Polluted")
        if "Unpolluted" in state_counts:
            fig_state.add_bar(x=state_counts[state_col], y=state_counts["Unpolluted"], name="Unpolluted")
        fig_state.update_layout(barmode="group", template="plotly_dark", title="Statewise Distribution", xaxis_tickangle=-45)
    else:
        fig_state = None

    # --- Enhanced Temporal Trend (species over time) ---
    temporal_fig = None
    if "date" in df.columns and species_cols:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.to_period("M").astype(str)
        fig_temp = go.Figure()
        colors = px.colors.qualitative.Plotly

        for i, sp in enumerate(species_cols):
            for label, line_type in [("Polluted", "solid"), ("Unpolluted", "dot")]:
                temp = df[df["label"] == label].groupby("month")[sp].mean().reset_index()
                fig_temp.add_trace(go.Scatter(
                    x=temp["month"],
                    y=temp[sp],
                    mode="lines+markers",
                    name=f"{sp} - {label}",
                    line=dict(dash=line_type, color=colors[i % len(colors)]),
                ))
        fig_temp.update_layout(template="plotly_dark", title="Temporal Trend of Top 8 Species", xaxis_tickangle=-45)
        temporal_fig = fig_temp

    # --- Correlation Heatmap ---
    corr_fig = None
    if species_cols and env_cols:
        corr_data = df[species_cols + env_cols].corr()
        corr_fig = px.imshow(
            corr_data,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap (Species vs Environment)",
            template="plotly_dark"
        )

 


    summary = {
        "total": len(df),
        "polluted": int((df["label"] == "Polluted").sum()),
        "unpolluted": int((df["label"] == "Unpolluted").sum()),
        
    }

    return summary, (fig_imp, fig_env, fig_species, fig_state, temporal_fig, corr_fig)



# === Gradio function ===
def run_pipeline_and_report(uploaded_csv, threshold):
    try:
        if uploaded_csv is None:
            return "Upload a CSV first.", None, None, None, None, None, None, None, None

        df = pd.read_csv(uploaded_csv.name)
        summary, figs = compute_analysis(df, threshold)
        fig_imp, fig_env, fig_species, fig_state, temporal_fig, corr_fig = figs



        try:
            pdf_path = generate_report(df)
        except TypeError:
            tmp = "tmp.csv"
            df.to_csv(tmp, index=False)
            pdf_path = generate_report(tmp)

        metrics = f"""
        **Total samples:** {summary['total']}  
        **Polluted:** {summary['polluted']}  
        **Unpolluted:** {summary['unpolluted']}  
        """
        return metrics, fig_imp, fig_env, fig_species, fig_state, temporal_fig, corr_fig, pdf_path
    except Exception as e:
        return f"❌ Error: {e}", None, None, None, None, None, None, None, None


# === Gradio Interface ===
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo:
    gr.Markdown("<h2 style='text-align:center;color:#00ff99'>COMPARATIVE ANALYSIS OF Pseudomonas IN POLLUTED AND UNPOLLUTED SOILS</h2>")

    with gr.Tab("Run Analysis"):
        input_csv = gr.File(label="Upload your dataset (CSV)")
        threshold = gr.Slider(0.0, 1.0, 0.3, step=0.01, label="Pollution threshold")
        run_btn = gr.Button("Run Analysis & Generate Report")

        status = gr.Markdown()
        feat_plot = gr.Plot()
        env_plot = gr.Plot()
        species_plot = gr.Plot()
        state_plot = gr.Plot()
        temporal_plot = gr.Plot()
        corr_plot = gr.Plot()
        pdf_output = gr.File(label="Download Generated Report")

        run_btn.click(
            fn=run_pipeline_and_report,
            inputs=[input_csv, threshold],
            outputs=[status, feat_plot, env_plot, species_plot, state_plot, temporal_plot, corr_plot, pdf_output]
        )

demo.launch(share=True)
