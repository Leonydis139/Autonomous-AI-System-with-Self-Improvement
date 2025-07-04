import streamlit as st
import pandas as pd
import plotly.express as px

def render_custom_analysis():
    st.header("🔍 Custom Analysis")
    st.subheader("Upload your own dataset for analysis")

    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload your dataset for custom analysis"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
            st.subheader("Data Preview")
            st.dataframe(df.head())
            st.subheader("Basic Statistics")
            st.write(df.describe())
            selected_columns = st.multiselect(
                "Select columns for analysis",
                df.columns
            )
            if selected_columns:
                st.subheader("Correlation Matrix")
                corr = df[selected_columns].corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                st.subheader("Distribution Plots")
                for col in selected_columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fig_dist = px.histogram(df, x=col, title=f"Distribution of {col}")
                        st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("Please upload a CSV file to get started")
