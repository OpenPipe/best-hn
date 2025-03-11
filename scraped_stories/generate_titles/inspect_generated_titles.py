#!/usr/bin/env python
"""
Streamlit app to visualize benchmark results from benchmark_models.py

To run this script: `echo inspect_generated_titles.py | entr -rs "uv run streamlit run inspect_generated_titles.py"`
"""

import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Page setup
st.set_page_config(
    page_title="HN Title Generator Benchmark", page_icon="📊", layout="wide"
)


# Load benchmark results
@st.cache_data
def load_benchmark_data():
    # Construct the path to benchmark_results.json in the same directory as this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "benchmark_results.json")
    with open(file_path, "r") as f:
        return json.load(f)


# App title
st.title("Hacker News Title Generator Benchmark")
st.markdown(
    "This app visualizes the performance of different models for generating Hacker News titles."
)

# Load data
try:
    benchmark_data = load_benchmark_data()
    st.success(f"Successfully loaded benchmark data with {len(benchmark_data)} stories")

    # Sort options at the top of the page
    sort_options = ["Original Score (High to Low)", "Original Score (Low to High)"]
    sort_by = st.selectbox("Sort Stories By", sort_options)

    # Process and sort data
    stories_data = []
    for story_id, story in benchmark_data.items():
        stories_data.append(
            {
                "id": story_id,
                "original_title": story["original_title"],
                "original_score": story.get("original_score", 0),
                "rm_score_original": story.get("rm_score_original", 0),
                "content": story["scraped_body"],
                "model_results": story.get("model_results", {}),
            }
        )

    # Calculate average scores by model
    st.header("Average Score by Model")

    # Collect all scores
    model_scores = {"Original": []}

    for story in stories_data:
        # Add original score
        if "rm_score_original" in story and story["rm_score_original"] is not None:
            model_scores["Original"].append(story["rm_score_original"])

        # Add model scores
        for model, result in story.get("model_results", {}).items():
            if model not in model_scores:
                model_scores[model] = []

            score = result.get("score")
            if score is not None:
                model_scores[model].append(score)

    # Calculate averages
    avg_scores = {}
    for model, scores in model_scores.items():
        if scores:  # Only calculate if we have scores
            avg_scores[model] = sum(scores) / len(scores)

    # Create DataFrame for visualization
    if avg_scores:
        avg_df = pd.DataFrame(
            {
                "Model": list(avg_scores.keys()),
                "Average Score": list(avg_scores.values()),
            }
        )

        # Sort by average score
        avg_df = avg_df.sort_values("Average Score", ascending=False)

        # Create visualization
        col1, col2 = st.columns([3, 1])

        with col1:
            fig, ax = plt.figure(figsize=(10, 6)), plt.subplot()
            bars = ax.bar(avg_df["Model"], avg_df["Average Score"], color="skyblue")
            ax.set_ylabel("Average Score")
            ax.set_title("Average Score by Model")

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.dataframe(avg_df, hide_index=True)
            st.markdown("**Note:** Higher scores indicate better performance.")
    else:
        st.info("No score data available for visualization.")

    # Sort based on selection
    if sort_by == "Original Score (High to Low)":
        stories_data.sort(key=lambda x: x["original_score"], reverse=True)
    else:
        stories_data.sort(key=lambda x: x["original_score"])

    # Display stories
    st.header("Stories with Generated Titles")

    for i, story in enumerate(stories_data):
        with st.expander(
            f"**{story['original_title']}** (Original Score: {story['original_score']})"
        ):
            # Create two columns - one for article content, one for model results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Original Article")
                st.markdown(story["content"])

            with col2:
                st.subheader("Titles")
                # Show all model results
                model_results = story["model_results"]

                if model_results:
                    # Create a DataFrame for the model results
                    results_data = []

                    # Add original score to the results
                    results_data.append(
                        {
                            "Model": "Original",
                            "Score": story["rm_score_original"],
                            "Title": story["original_title"],
                        }
                    )
                    for model, result in model_results.items():
                        results_data.append(
                            {
                                "Model": model,
                                "Score": result.get("score", -1),
                                "Title": result.get("generated_title", ""),
                            }
                        )

                    # Sort by score
                    results_df = pd.DataFrame(results_data)
                    if not results_df.empty and "Score" in results_df.columns:
                        results_df = results_df.sort_values(by="Score", ascending=False)

                    # Display as a table that fills the column width
                    st.dataframe(
                        results_df,
                        hide_index=True,
                    )
                else:
                    st.info("No model results available for this story.")

except Exception as e:
    st.error(f"Error loading benchmark data: {e}")
    st.info(
        "Make sure you've run benchmark_models.py to generate benchmark_results.json"
    )
    st.exception(e)
