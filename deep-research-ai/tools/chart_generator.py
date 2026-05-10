from __future__ import annotations

import json
import re
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px


def extract_chart_blocks(markdown_report: str) -> list[dict]:
    """Extract JSON chart blocks from markdown report."""
    pattern = r'```chart\s*\n(.*?)\n```'
    matches = re.findall(pattern, markdown_report, re.DOTALL)
    charts = []
    for match in matches:
        try:
            chart_data = json.loads(match.strip())
            charts.append(chart_data)
        except json.JSONDecodeError:
            continue
    return charts


def generate_chart(chart_spec: dict, output_dir: str | Path = "outputs") -> str:
    """Generate a Plotly chart from specification and save as HTML.
    
    Args:
        chart_spec: Chart specification with type, title, labels, values, etc.
        output_dir: Directory to save chart HTML files
        
    Returns:
        HTML embed code for the chart
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chart_type = chart_spec.get("type", "bar")
    title = chart_spec.get("title", "Chart")
    x_label = chart_spec.get("x_label", "X")
    y_label = chart_spec.get("y_label", "Y")
    labels = chart_spec.get("labels", [])
    values = chart_spec.get("values", [])
    
    # Create filename from title
    safe_title = re.sub(r'[^a-z0-9]+', '_', title.lower())
    chart_filename = f"{safe_title}.html"
    chart_path = output_dir / chart_filename
    
    # Detect if values are multi-series (list of dicts with name/values)
    is_multi_series = (
        isinstance(values, list) and 
        len(values) > 0 and 
        isinstance(values[0], dict) and 
        "name" in values[0] and 
        "values" in values[0]
    )
    
    # Generate appropriate chart based on type
    if chart_type == "bar":
        if is_multi_series:
            # Multi-series bar chart
            fig = go.Figure()
            for series in values:
                fig.add_trace(go.Bar(
                    x=labels,
                    y=series.get("values", []),
                    name=series.get("name", ""),
                ))
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=500,
                template="plotly_white",
                hovermode="x unified",
                barmode="group",
            )
        else:
            # Single-series bar chart
            fig = go.Figure(
                data=[go.Bar(x=labels, y=values, marker_color="steelblue")],
                layout=go.Layout(
                    title=title,
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    height=500,
                    template="plotly_white",
                    hovermode="x unified",
                )
            )
    elif chart_type == "line":
        if is_multi_series:
            # Multi-series line chart
            fig = go.Figure()
            for series in values:
                fig.add_trace(go.Scatter(
                    x=labels,
                    y=series.get("values", []),
                    mode="lines+markers",
                    name=series.get("name", ""),
                ))
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=500,
                template="plotly_white",
                hovermode="x unified",
            )
        else:
            # Single-series line chart
            fig = go.Figure(
                data=[go.Scatter(x=labels, y=values, mode="lines+markers", line=dict(color="steelblue"))],
                layout=go.Layout(
                    title=title,
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    height=500,
                    template="plotly_white",
                    hovermode="x unified",
                )
            )
    elif chart_type == "pie":
        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values)],
            layout=go.Layout(title=title, height=500, template="plotly_white")
        )
    elif chart_type == "scatter":
        if is_multi_series:
            # Multi-series scatter
            fig = go.Figure()
            for series in values:
                fig.add_trace(go.Scatter(
                    x=labels,
                    y=series.get("values", []),
                    mode="markers",
                    name=series.get("name", ""),
                    marker=dict(size=8)
                ))
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=500,
                template="plotly_white",
                hovermode="closest",
            )
        else:
            # Single-series scatter
            fig = go.Figure(
                data=[go.Scatter(x=labels, y=values, mode="markers", marker=dict(size=8, color="steelblue"))],
                layout=go.Layout(
                    title=title,
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    height=500,
                    template="plotly_white",
                    hovermode="closest",
                )
            )
    elif chart_type == "radar":
        if is_multi_series:
            # Multi-series radar chart
            fig = go.Figure()
            for series in values:
                fig.add_trace(go.Scatterpolar(
                    r=series.get("values", []),
                    theta=labels,
                    fill="toself",
                    name=series.get("name", ""),
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title=title,
                height=600,
                template="plotly_white",
            )
        else:
            # Single-series radar (shouldn't happen but handle it)
            fig = go.Figure(
                data=[go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill="toself",
                    name="Data"
                )],
                layout=go.Layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title=title,
                    height=600,
                    template="plotly_white",
                )
            )
    else:
        # Default to bar
        if is_multi_series:
            fig = go.Figure()
            for series in values:
                fig.add_trace(go.Bar(
                    x=labels,
                    y=series.get("values", []),
                    name=series.get("name", ""),
                ))
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=500,
                template="plotly_white",
            )
        else:
            fig = go.Figure(
                data=[go.Bar(x=labels, y=values, marker_color="steelblue")],
                layout=go.Layout(
                    title=title,
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    height=500,
                    template="plotly_white",
                )
            )
    
    # Save as HTML
    fig.write_html(str(chart_path))
    
    # Return markdown embed code
    return f'<iframe src="file://{chart_path.absolute()}" width="100%" height="600" frameborder="0"></iframe>'


def embed_charts_in_report(markdown_report: str, output_dir: str | Path = "outputs") -> str:
    """Replace chart JSON blocks with actual embedded charts."""
    charts = extract_chart_blocks(markdown_report)
    
    if not charts:
        print(f"[CHART DEBUG] No charts found in report")
        return markdown_report
    
    print(f"[CHART DEBUG] Found {len(charts)} charts to generate")
    
    # Generate all charts
    chart_embeds = []
    for idx, chart_spec in enumerate(charts):
        try:
            print(f"[CHART DEBUG] Generating chart {idx+1}: {chart_spec.get('title', 'Untitled')}")
            embed_code = generate_chart(chart_spec, output_dir)
            chart_embeds.append(embed_code)
            print(f"[CHART DEBUG] Successfully generated chart {idx+1}")
        except Exception as e:
            print(f"[CHART DEBUG] Error generating chart {idx+1}: {e}")
            import traceback
            traceback.print_exc()
            chart_embeds.append(f"<!-- Chart generation failed: {e} -->")
    
    # Replace chart blocks with embeds
    updated_report = markdown_report
    pattern = r'```chart\s*\n(.*?)\n```'
    
    for embed_code in chart_embeds:
        # Replace one at a time to preserve order
        updated_report = re.sub(pattern, f"\n{embed_code}\n", updated_report, count=1, flags=re.DOTALL)
    
    print(f"[CHART DEBUG] Report processing complete")
    return updated_report
