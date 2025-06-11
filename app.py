import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

# Load datasets
@st.cache_data
def load_data():
    df = pd.read_csv("Complete Dataset WIF3009.csv")
    centrality = pd.read_csv("influencer_centrality_metrics.csv")
    edges = pd.read_csv("interaction_edges.csv")

    # Patch missing Engagement Rate if needed
    if 'Engagement Rate' not in df.columns:
        df['Engagement Rate'] = (
            (df['No of Likes'].astype(float) / df['Video Views'].astype(float) +
             df['No of Comments'].astype(float) / df['Video Views'].astype(float)) / 2
        ).fillna(0)

    return df, centrality, edges

df, centrality_df, edges_df = load_data()

# Try loading additional visual CSVs if present
try:
    imp_df = pd.read_csv("feature_importance.csv")
except:
    imp_df = None

try:
    svd_keywords = pd.read_csv("svd_topic_keywords.csv")
except:
    svd_keywords = None

try:
    top_keywords = pd.read_csv("top_nonname_keywords.csv")
except:
    top_keywords = None

# --- Sidebar Navigation --- #
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview", "Network Analysis", "Influence Prediction",
    "Content & Sentiment", "Dataset Explorer", "Summary & Recommendations"
])

# --- Page: Overview --- #
if page == "Overview":
    st.title("Social Network Analysis of YouTubers")

    st.markdown("""
    This dashboard presents an AI-driven social network analysis focused on **YouTube influencers**. 

    Our project explores the connections, influence dynamics, and content patterns among content creators, leveraging Social Network Analysis (SNA) and machine learning models.

    ### 🧩 What This Dashboard Covers
    - Key influencer rankings based on PageRank and network centrality
    - AI-based predictions for future engagement and influence
    - Topic and sentiment trends from video content
    - Recommendations for content strategy and outreach
    """)

    st.subheader("📊 Summary Statistics")
    total_influencers = len(df["channel_name"].unique()) if 'channel_name' in df.columns else len(df["Creator Name"].unique())
    st.metric("Total Influencers Analyzed", total_influencers)
    st.metric("Total Videos", len(df))

    avg_engagement = df['Engagement Rate'].mean()
    st.metric("Average Engagement Rate", f"{avg_engagement:.4f}")

    st.markdown("""
    - An **engagement rate** of ~{:.2f}% is considered **moderate**. For YouTube, typical engagement rates range from 1%–5%. This means YouTubers are performing reasonably well in terms of likes and comments per view.
    - Engagement rate is calculated as the **average of likes-per-view and comments-per-view**, providing a normalized metric for cross-channel comparison.
    """.format(avg_engagement * 100))

    st.info("This study uses real data extracted from public YouTube channels, enhanced with AI models to interpret content quality, influence trends, and viewer behavior.")

# --- Page: Network Analysis --- #
elif page == "Network Analysis":
    st.title("Influencer Network Analysis")

    st.markdown("""
    This section analyzes the influencer network based on real group and collaboration data, derived from shared videos and known partnerships.

    Key insights include:
    - Centrality rankings (e.g., PageRank, Betweenness)
    - Interaction edges (direct + same group)
    - Community detection (Louvain)
    - Weak tie analysis
    - Ego networks for key influencers
    """)

    st.subheader("Top Influencers by PageRank")
    st.markdown("""
    The PageRank algorithm ranks influencers based on both the number and quality of connections they receive from others.
    - A high PageRank suggests that an influencer is not only popular, but also **frequently cited by other well-connected influencers**.
    - This makes them excellent candidates for **amplifying campaigns** or driving visibility.
    """)
    st.dataframe(centrality_df.head(10))

    st.subheader("Interaction Edges (Sample)")
    st.markdown("""
    This table displays the types of interactions detected between influencers:
    - **direct_mention**: one influencer explicitly collaborated or referenced another.
    - **same_group**: members belonging to a shared collective or group.
    The weight indicates how often this connection was detected.
    """)
    st.dataframe(edges_df.head(10))

    st.subheader("PageRank Distribution")
    st.markdown("""
    This histogram shows the spread of PageRank scores across all influencers.
    - A **long tail** distribution may indicate a few dominant hubs.
    - A **flat** distribution suggests a decentralized network.
    This helps identify how influence is distributed in the YouTube scene.
    """)
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(centrality_df['PageRank'], bins=20, kde=False, color='skyblue', ax=ax)
        ax.set_title("PageRank Distribution Across Influencers")
        ax.set_xlabel("PageRank Value")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    except Exception:
        st.warning("Unable to generate PageRank histogram.")

    st.subheader("Weak Ties in the Network")
    st.markdown("""
    Weak ties refer to **infrequent or low-weight connections**.
    - While not strong in volume, they often serve as **bridges between communities**.
    - These ties are critical for spreading new content across otherwise disconnected influencer clusters.
    This visualization highlights the bottom 5% of edge weights in red.
    """)
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.DiGraph()
        for _, row in edges_df.iterrows():
            G.add_edge(row['source'], row['target'], weight=row['weight'])
        threshold = edges_df['weight'].quantile(0.05)
        weak_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] <= threshold]
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray', ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=weak_edges, edge_color='red', ax=ax)
        ax.set_title("Weak Ties (Bottom 5% Edge Weights)")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Weak tie graph failed: {e}")

    st.subheader("Top Influencer Ego Networks")
    st.markdown("""
    These ego networks show the **direct neighborhood** of each top influencer:
    - Each diagram includes only the influencer and their immediate connections.
    - Helps visualize **influence reach**, **connection density**, and potential **collaboration clusters**.
    These visuals can guide outreach strategy or content partnerships.
    """)
    try:
        top_influencers = centrality_df.head(3)['Influencer']
        G_full = nx.DiGraph()
        for _, row in edges_df.iterrows():
            G_full.add_edge(row['source'], row['target'], weight=row['weight'])
        for influencer in top_influencers:
            ego = nx.ego_graph(G_full, influencer, radius=1)
            fig, ax = plt.subplots(figsize=(6, 4))
            pos = nx.spring_layout(ego, seed=42)
            nx.draw(ego, pos, with_labels=True, node_size=300, edge_color='gray', ax=ax)
            ax.set_title(f"Ego Network of {influencer}")
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Ego network visualization failed: {e}")

# --- Page: Influence Prediction --- #
elif page == "Influence Prediction":
    st.title("Predicting Engagement Rate")

    st.markdown("""
    This tool uses a simplified model to estimate an influencer’s future engagement rate based on video and creator features. The prediction is based on your inputs for duration, sentiment, hashtags, and other metadata.

    Typical engagement rates for YouTube influencers range between **1% to 5%**. Use this tool to see how content features might affect expected performance.
    """)

    duration = st.slider("Video Duration (minutes)", 0, 60, 5)
    polarity = st.slider("Sentiment Polarity", -1.0, 1.0, 0.1)
    subjectivity = st.slider("Sentiment Subjectivity", 0.0, 1.0, 0.5)
    hashtags = st.slider("Hashtag Count", 0, 20, 3)
    subtitle = st.selectbox("Subtitle Available", ["Yes", "No"])

    dummy_engagement = (0.1 * duration + 0.5 * polarity + 0.3 * hashtags) / 100
    st.success(f"Predicted Engagement Rate: {dummy_engagement:.4f}")

    st.markdown("### Interpretation")
    if dummy_engagement < 0.01:
        st.warning("This predicted rate is below the typical range for YouTube. It may indicate low engagement.")
    elif dummy_engagement < 0.05:
        st.info("This predicted rate is within the common range (1%–5%) for YouTube videos. Moderate performance expected.")
    else:
        st.success("This predicted rate is higher than average — potentially strong engagement!")

    st.subheader("Top 20 Feature Importances (XGBoost)")
    st.markdown("""
    This chart shows which features were most important in predicting engagement rate using the XGBoost regression model.

    - **SVD_\*** features represent compressed content themes extracted from video titles and descriptions using **Singular Value Decomposition** on a TF-IDF matrix. These latent dimensions reflect style, topic, or structure of content that may impact viewer engagement.
    - Other features like **Sentiment Polarity**, **Hashtag Count**, and **Duration** reflect specific characteristics of individual videos.
    - **Creator-level features** like `No of Playlist`, `Total Subscribers`, or `Gender` provide broader context on the influencer.

    The higher the bar, the more influence that feature had on the model's predictions.
    """)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

# --- Page: Content & Sentiment --- #
elif page == "Content & Sentiment":
    st.title("Content & Sentiment Analysis")
    
    # Load the richer dataset for this page only
    try:
        df = pd.read_csv("final_dataset.csv")
    except:
        st.error("final_dataset.csv not found. Please upload it.")

    st.subheader("Sentiment Polarity Distribution")
    st.markdown("""
    This chart shows the distribution of **sentiment polarity** across video content:
    - A score close to **1.0** indicates very **positive** sentiment.
    - A score close to **-1.0** reflects strong **negative** sentiment.
    - A value near **0.0** means the content is **neutral**.

    The majority of influencers tend to publish content that is either neutral or positive in tone — a trend often correlated with higher engagement.
    """)
    if 'Sentiment Polarity' in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df['Sentiment Polarity'], bins=30, ax=ax, color="skyblue")
        st.pyplot(fig)

    st.subheader("Upload Timing Patterns")
    st.markdown("""
    This heatmap shows how **upload day and hour** correlate with engagement rate.
    - Rows represent the day of the week (0 = Monday, 6 = Sunday)
    - Columns represent the hour of upload (0–23)

    Brighter areas suggest higher engagement levels — which can help determine optimal posting times for YouTubers.
    """)
    if 'Upload DayOfWeek' in df.columns and 'Upload Hour' in df.columns:
        heatmap_data = df.pivot_table(index='Upload DayOfWeek', columns='Upload Hour',
                                      values='Engagement Rate', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    st.subheader("Top Keywords by SVD Component")
    st.markdown("""
    The keywords shown below are derived from **SVD components** (Singular Value Decomposition) based on TF-IDF of video titles and descriptions.

    Each component represents a **latent topic dimension** that groups semantically related words. These themes help reveal common content structures or focus areas that influence engagement.
    """)
    if svd_keywords is not None:
        top_components = svd_keywords["Component"].unique()[:3]
        for component in top_components:
            st.markdown(f"**{component}**")
            subset = svd_keywords[svd_keywords["Component"] == component]
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(data=subset, x="Score", y="Word", ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Keyword component data not available.")

    st.subheader("Top Non-Name Keywords (Content Themes)")
    st.markdown("""
    This list displays the most frequent meaningful keywords used across influencer content (excluding names).
    - These reflect recurring themes or focus areas.
    - High-frequency keywords may correlate with popular or trending video types.

    This can guide content creators in shaping their message or targeting topics that resonate with their audience.
    """)
    if top_keywords is not None:
        st.dataframe(top_keywords.head(20))
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=top_keywords.head(20), x="Count", y="Keyword", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Top keyword summary not available.")

# --- Page: Summary & Recommendations --- #
elif page == "Summary & Recommendations":
    st.title("📌 Summary & Strategic Recommendations")

    st.markdown("""
    This project applied Social Network Analysis (SNA) and machine learning to analyze the YouTube influencer landscape.

    ### 🔍 Key Insights
    - **Network Analysis** revealed that a few influencers play highly central roles in collaboration patterns, as seen through PageRank and ego networks.
    - **Weak ties**, while rare, may serve as valuable bridges for outreach into new communities.
    - **Content themes** extracted via SVD highlighted recurring topics and styles that drive engagement.
    - **Engagement predictors** such as sentiment polarity, hashtags, and video duration were top contributors to the model.

    ### 📈 Data Sources
    - Extracted public data from YouTube channels
    - Derived sentiment and topics using NLP techniques
    - Created network graphs using collaboration and group affiliations

    ### ✅ Recommendations
    - **Target high PageRank influencers** for brand campaigns—they’re structurally important in the network.
    - **Optimize content** by aligning with dominant SVD topics and top-performing keywords.
    - **Post at high-engagement times** based on upload timing analysis (if available).
    - **Leverage weak ties** to enter new subcommunities and diversify visibility.

    ### 📚 Tools Used
    - `pandas`, `networkx`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`
    - Streamlit for interactive reporting

    ---
    _This project was developed for the WIF3009 course as a capstone in influencer analytics._
    """)

# --- Page: Dataset Explorer --- #
elif page == "Dataset Explorer":
    st.title("Complete Dataset Viewer")

    st.dataframe(df.head(100))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Dataset", csv, "WIF3009_Influencer_Data.csv", "text/csv")