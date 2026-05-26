import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.utils.config_loader import load_config, logger

# Initialize VADER lexicon if not already done
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    logger.info("VADER lexicon not found, downloading...")
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

# High fidelity Mock news feeds and tweets for different sectors
MOCK_FEED = {
    "AI & DeepTech": [
        "Indian AI startups raise massive funding as deeptech innovation hits all-time high.",
        "New regulatory concerns emerge over synthetic media generation in AI sector.",
        "Breakthrough: Local AI startup builds a bilingual LLM optimizing enterprise workflows.",
        "AI chip manufacturing plans in South India spark enthusiasm among tech entrepreneurs.",
        "Talent crunch: High costs of AI engineers holding back early-stage deeptech companies."
    ],
    "FinTech": [
        "RBI introduces new regulatory guidelines for digital lending, boosting credit access safety.",
        "Unified Payments Interface (UPI) logs record-breaking transactions; FinTechs celebrate.",
        "FinTech startups see drop in valuation amidst tightening venture capital funding.",
        "Cybersecurity breach at a leading digital wallet raises red flags across banking sector.",
        "Innovative Neo-bank partners with public sector banks to offer seamless rural microfinance."
    ],
    "EdTech": [
        "EdTech rebound: Startups shifting focus from aggressive sales to high-quality hybrid classrooms.",
        "Reports indicate mass layoffs at prominent EdTech giants amidst structural shifts.",
        "Free coding classes for government school students launched by a non-profit EdTech platform.",
        "Parents express concern over rising costs of subscription models in K-12 learning apps.",
        "EdTech sector adopts AI-driven personalized tutoring, showing 40% improvement in student retention."
    ],
    "E-Commerce & Retail": [
        "Quick commerce sector explodes with Zepto, Blinkit delivering groceries in under 8 minutes.",
        "Local kirana stores express discontent over aggressive discounting by e-commerce giants.",
        "Sustainable e-commerce brand secures series B funding for zero-waste packaging supplies.",
        "Supply chain bottleneck during festive seasons creates logistics strain for small retailers.",
        "ONDC (Open Network for Digital Commerce) enables 50,000 small merchants to list catalogs."
    ],
    "CleanTech & Green Energy": [
        "Government subsidies on EV infrastructure create boom for battery swapping startups.",
        "Solar grid startups report record adoption in Western and Southern Indian states.",
        "CleanTech startup pioneers bio-degradable crop-waste plastics to tackle pollution.",
        "Venture capitalists show massive appetite for green hydrogen and battery recycling technologies.",
        "Regulatory delays in offshore wind farm certifications cause minor setbacks."
    ],
    "HealthTech & BioTech": [
        "Telemedicine apps reach 90% of tier-2 and tier-3 cities, expanding quality diagnostic access.",
        "BioTech startup successfully prints India's first 3D bio-compatible skin tissue matrix.",
        "Security concerns voiced over patient data privacy and cloud breaches on health portals.",
        "New affordable cancer detection kits developed by startup receives ICMR approval.",
        "Doctors express skepticism over AI diagnostic accuracy without rigorous clinical trials."
    ],
    "Logistics & Mobility": [
        "EV-only delivery fleet startups sign mega-deals with quick commerce companies.",
        "High fuel prices and highway tolls squeeze margins for long-haul logistics startups.",
        "Indian drone logistics startup gets DGCA approval for rural medicine delivery trials.",
        "AI-enabled route optimization reduces urban delivery carbon footprint by 25%.",
        "Gig workers call for fair wages and standard safety protections from ride-hailing startups."
    ]
}

def analyze_text(text):
    """Analyzes text using VADER and returns positive, negative, neutral and compound scores."""
    if not text:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0, "sentiment": "Neutral"}
        
    scores = sia.polarity_scores(text)
    
    # Determine overall sentiment label
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    scores["sentiment"] = sentiment
    return scores

def get_sector_sentiment_stats(sector):
    """Analyzes mock news feed for a given sector and aggregates sentiment results."""
    feeds = MOCK_FEED.get(sector, [
        f"Indian startup ecosystem is growing in {sector} sector.",
        f"New startups emerge in {sector} despite funding winter.",
        f"Startups face competition and operational challenges in {sector}."
    ])
    
    pos_sum, neg_sum, neu_sum, comp_sum = 0, 0, 0, 0
    analyzed_feeds = []
    
    for item in feeds:
        score = analyze_text(item)
        pos_sum += score["pos"]
        neg_sum += score["neg"]
        neu_sum += score["neu"]
        comp_sum += score["compound"]
        
        analyzed_feeds.append({
            "text": item,
            "sentiment": score["sentiment"],
            "score": score["compound"]
        })
        
    n = len(feeds)
    avg_scores = {
        "pos": pos_sum / n,
        "neg": neg_sum / n,
        "neu": neu_sum / n,
        "compound": comp_sum / n
    }
    
    if avg_scores["compound"] >= 0.05:
        label = "Positive"
    elif avg_scores["compound"] <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
        
    avg_scores["sentiment"] = label
    
    return avg_scores, analyzed_feeds

if __name__ == "__main__":
    test_text = "ZEPTO is doing exceptionally well in quick commerce, expanding at an incredible rate!"
    print(f"Text: {test_text}")
    print("Scores:", analyze_text(test_text))
    
    avg, feeds = get_sector_sentiment_stats("FinTech")
    print("\nFinTech Sector Average Sentiment:")
    print(avg)
    print("FinTech Sample Feed Item:")
    print(feeds[0])
