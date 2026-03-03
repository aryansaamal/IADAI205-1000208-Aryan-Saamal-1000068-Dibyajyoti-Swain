
# ✈️ TravelAI Premium

## AI Cultural Tourism Insights & Engagement Platform

An adaptive, AI-driven tourism system designed to generate **meaningful, personalized, and culturally immersive itineraries**, supported by intelligent destination recommendations, multilingual output, and feedback-based improvement.

---

**Acknowledgements**

I would like to express my sincere gratitude to my CRS-AI mentor for her continuous guidance and constructive feedback throughout the development of this capstone project, TravelAI Premium – AI Cultural Tourism Insights & Engagement Platform. Her support helped transform the initial idea into a structured, deployable AI system.

I also acknowledge the IBCP framework for encouraging applied learning, critical thinking, and real-world problem solving. The flexibility of the program allowed this project to evolve through research, experimentation, testing, and deployment.

Finally, I extend my appreciation to the developers and contributors of the open-source tools and APIs used in this project, which made the technical implementation possible.


# 🌍 Project Foundation
## 🔎 Understanding the Problem
Before building the system, we studied major travel platforms such as MakeMyTrip, TripAdvisor, Agoda, Airbnb, and Skyscanner.

We observed one key gap:

While these platforms are strong in bookings and listings, they rarely focus on meaningful itinerary creation based on personal preferences, cultural depth, travel intent, and language customization.

Travelers spend hours switching between platforms, comparing information, and still end up with generic travel plans.

TravelAI Premium was built to solve this — by focusing on meaning, personalization, intelligence, and adaptability.
---

# 🎯 Core Objective

We often noticed that
Most of the apps answer:
> “Where can I go?”

But..
Very few answer:
> “Why is this meaningful for me?”

**That became our core focus.**

To build an AI platform that:

* Understands user preferences
* Recommends destinations intelligently
* Generates culturally meaningful itineraries
* Supports multilingual outputs
* Estimates cost and travel feasibility
* Assists locally with actionable insights
* Learns from collective feedback

---


# 🗓️ Week-by-Week Development Process

---

## ✅ Week 1 – Data Understanding & Exploration

* Loaded datasets into Google Colab
* Performed exploratory data analysis
* Generated bar charts & visual patterns
* Understood tourism trends
* Studied budget patterns & climate behavior

<img width="1408" height="1239" alt="image" src="https://github.com/user-attachments/assets/9d4997f4-b923-4b27-b70a-692e854fe74d" />

<img width="1677" height="1100" alt="image" src="https://github.com/user-attachments/assets/1e8e1ebd-1707-44a0-b39a-36c5c197cbc2" />

Goal: Understand how tourism data behaves before building logic.

---

## ✅ Week 2 – Data Cleaning & Feature Engineering

Performed:

* Data cleaning
* Column normalization
* Feature structuring
* Integration of multiple datasets

Final output:

**master_dataset_week2.csv**

* 18,520 rows
* 31 columns

<img width="1332" height="1009" alt="image" src="https://github.com/user-attachments/assets/9936e04b-3576-4b18-8522-a7dd89fd9f09" />

This became the core engine dataset.

It includes:

* Destination metadata
* Budget levels
* Cultural scores
* Climate labels
* Coordinates
* Experience dimensions

Week 2 laid the foundation for personalization.

---

## ✅ Week 3 – Core Recommendation Engine (Streamlit Start)

* Began Streamlit application
* Built traveler profile system
* Created multi-factor scoring algorithm
* Enabled country-to-country intelligent matching

<img width="1573" height="559" alt="image" src="https://github.com/user-attachments/assets/00b01d87-a14c-452e-8b80-581f85ecbab6" />

This is where the system became interactive.

---

## ✅ Week 4 – Challenges & Redirection

Challenge:
The dataset did not contain images.
A tourism platform without visuals feels incomplete.

### Solution Innovation

Instead of relying on static image datasets:

* Integrated OpenRouter API
* Integrated Google Maps using latitude & longitude
* Redirected users to live Google searches for:

  * Attractions
  * Restaurants
  * Hotels
  * Images
  * Transport

The platform became dynamically connected to real-world data.

<img width="1698" height="659" alt="image" src="https://github.com/user-attachments/assets/370ec046-1805-4d0c-83d7-89851b2297f7" />

---

## ✅ Week 5 – Smart Explorer & Visual Intelligence

Implemented:

* Google Maps embed view
* Location-based redirection
* Smart external research links
* Cultural scoring visualization

Users now understood:

> Why the system selected this destination.

Transparency added trust.
<img width="1239" height="680" alt="image" src="https://github.com/user-attachments/assets/022726a6-aeee-4de7-bc1f-5039ba874c89" />
<img width="1243" height="672" alt="image" src="https://github.com/user-attachments/assets/25f5639d-c173-4bd0-9589-fd8c0b20c68e" />

---

## ✅ Week 6 – AI Itinerary Generator

Integrated GPT-based itinerary engine.

Personalized by:

* Age group
* Interests
* Budget
* Language
* Trip duration

Generated:
Structured, multi-day cultural travel master plans.

<img width="1103" height="790" alt="Screenshot 2026-03-02 123752" src="https://github.com/user-attachments/assets/b807aac0-031e-4433-a95f-82623cc0fb30" />
<img width="1107" height="816" alt="Screenshot 2026-03-02 123804" src="https://github.com/user-attachments/assets/9062df04-1140-4697-a7c7-0b86efbe8027" />

---

## ✅ Week 7 – Professional PDF Export

Built:

* Downloadable structured travel guide
* Personalized metadata
* User rating integration
* Clean formatting

User receives a professional itinerary document.
<img width="1788" height="671" alt="Screenshot 2026-03-02 123917" src="https://github.com/user-attachments/assets/f83ceeb4-11f7-4fcd-a2cf-6fdb292bac98" />

<img width="1183" height="672" alt="Screenshot 2026-03-02 123938" src="https://github.com/user-attachments/assets/bfe8ba16-264c-4923-bdd5-6f34aabf7451" />

---

## ✅ Week 8 – Cinematic Travel Recap (GIF-Based Innovation)

Since no image dataset existed:

We built a dynamic animated GIF style video storytelling system.

Features:

* Destination intro
* Budget overview
* Climate snapshot
* Trip schedule slides
* Cultural tips
* Closing recap

This replaced static images with a narrative visual journey.
Innovation under constraint.
<img width="1137" height="780" alt="image" src="https://github.com/user-attachments/assets/5139c204-250e-40ab-b011-ca6379a65a43" />

# Week 8 Main Integration--🤖 Hodophiler — The Core Intelligence Engine

The most dominant component of TravelAI Premium is the chatbot, **Hodophiler**.

The name comes from the word *hodophile*, meaning a person who loves to travel. Inspired by this, Hodophiler was designed to act as an intelligent travel companion rather than just a feature.

---

## 🌍 What Makes Hodophiler Powerful

Unlike other components that rely strictly on structured dataset logic, Hodophiler goes beyond the dataset.

It:

* Understands the selected destination, climate, budget, and user interests
* Provides real-time travel advice
* Suggests packing lists, local transport, food, attractions, and safety tips
* Offers cultural context and etiquette guidance
* Supports multilingual interactions
* Responds dynamically using live AI intelligence

It does not simply repeat stored data — it expands, explains, and contextualizes travel decisions.

---

## 🎯 Its Role in the System

If the recommendation engine selects the destination and the itinerary generator builds the plan, Hodophiler connects everything together.

It:
* Also Answers the questions like with whom you can travel, preffered Seasons, etc 
* Resolves user doubts instantly
* Enhances personalization
* Provides guidance beyond predefined logic
* Makes the platform interactive and human-like

Without Hodophiler, the system would function.
With Hodophiler, it becomes intelligent and conversational.

<img width="966" height="786" alt="Screenshot 2026-03-02 130144" src="https://github.com/user-attachments/assets/af34db1b-7663-4585-852a-bc1bcb010809" />
<img width="714" height="262" alt="Screenshot 2026-03-02 130227" src="https://github.com/user-attachments/assets/fae0f960-ce7e-4f05-b2ff-550e1186a6fd" />


## ✅ Week 9 – Feedback & Adaptive Learning Engine

Implemented:

* Feedback form
* Feature rating (Itinerary, Recommendation, Video, Chatbot)
* Language & season tracking
* Trend analytics dashboard
* Radar performance visualization
<img width="1099" height="780" alt="Screenshot 2026-03-02 124155" src="https://github.com/user-attachments/assets/9c3781e8-f138-4db3-945a-7a75573d99f7" />
<img width="1109" height="800" alt="Screenshot 2026-03-02 124216" src="https://github.com/user-attachments/assets/2a1cf652-87f5-4235-b762-01b3661c89a9" />

### Adaptive Boost Mechanism

Destinations with:

* ≥ 4.0 rating → +15 recommendation boost
* ≥ 3.0 rating → +5 boost

The system improves over time.

Closed-loop intelligent architecture.

---

## ✅ Week 10 – Integration & Deployment

Finalization included:

* Modular architecture cleanup
* Secure API key handling
* GitHub repository structuring
* requirements.txt integration
* README documentation
* Streamlit Cloud deployment
* Final system testing

TravelAI Premium became fully deployable.

**Links: **
Streamlit link - https://capstonedibyajyoti0018.streamlit.app/ 

---

# 🧠 System Architecture

User Input →
Scoring Algorithm →
Best Destination →
AI Itinerary Generator →
Google Maps & Explorer →
PDF Export →
Cinematic Recap →
Feedback Collection →
Adaptive Boost Engine

Closed adaptive AI tourism ecosystem.

---
**🚀 Innovations Implemented**

1️⃣ Meaning-Centric Itinerary Generation
Instead of generic travel lists, the system generates structured cultural master plans tailored to traveler intent.

2️⃣ Multi-Factor Weighted Decision Model
Destination selection is based on interest scores, budget compatibility, ratings, duration bonus, and adaptive feedback boosts.

3️⃣ Live Google Maps & Smart Research Redirection
Integrated dynamic Google Maps view and real-time research links for restaurants, hotels, attractions, and images.

4️⃣ Cinematic Recap Engine Under Data Constraints
In absence of image datasets, developed an animated storytelling engine that simulates a travel preview experience.

5️⃣ Active Hodophiler AI Chatbot
The chatbot is dynamic and up-to-date.
It does not depend only on the internal dataset.
It generates contextual travel advice, packing tips, local suggestions, and cultural insights using live language intelligence beyond stored data.

6️⃣ Adaptive Feedback Learning Mechanism
Highly rated destinations automatically receive scoring boosts, allowing the system to evolve with user behavior.

# ⚖️ Limitations

1. Feedback stored locally (not persistent cloud DB)
2. Rule-based weighted scoring (not deep ML model)
3. No direct booking integration
4. Dataset does not include media assets
5. Cinematic recap uses slide animation, not real footage AI

---

# 🔮 Future Scope

Planned enhancements:

* Cloud database (Firebase/Supabase)
* Real-time hotel & flight APIs
* Machine learning collaborative filtering
* Sentiment analysis on comments
* Climate API integration
* Tourism demand forecasting
* Mobile app version
* AR/VR immersive preview
* Real-time cost fluctuation integration

---

# 🎓 Conclusion

TravelAI Premium started from a research question:

> Why do travelers struggle to build meaningful itineraries?

It evolved into:

An adaptive AI-powered cultural tourism intelligence platform.

This project demonstrates:

* Data engineering
* Feature modeling
* Multi-factor scoring design
* AI language integration
* Human-centered personalization
* Visual storytelling innovation
* Feedback-driven adaptive systems
* Real-world deployment capability

---

# 💻 Tech Stack

* Python
* Streamlit
* OpenRouter API
* Pandas
* Plotly
* Matplotlib
* FPDF
* PIL

---
**(Repository of Aryan Saamal-1000208)**
# Capstone Project By,
Aryan Saamal & Dibyajyoti Swain
AI Cultural Tourism Insights & Engagement Platform
IBCP (CRS-Artificial Intelligence)


**Bibilography:**
Airbnb. (n.d.). *Airbnb: Vacation rentals, cabins, beach houses & more*. [https://www.airbnb.com](https://www.airbnb.com)

MakeMyTrip. (n.d.). *MakeMyTrip: Flights, hotels, holiday packages & more*. [https://www.makemytrip.com](https://www.makemytrip.com)

OpenRouter. (n.d.). *OpenRouter API documentation*. [https://openrouter.ai/docs](https://openrouter.ai/docs)

Plotly Technologies Inc. (n.d.). *Plotly Python documentation*. [https://plotly.com/python/](https://plotly.com/python/)

Streamlit Inc. (n.d.). *Streamlit documentation*. [https://docs.streamlit.io](https://docs.streamlit.io)

Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender systems handbook* (2nd ed.). Springer.

Xiang, Z., & Gretzel, U. (2010). Role of social media in online travel information search. *Tourism Management, 31*(2), 179–188. [https://doi.org/10.1016/j.tourman.2009.02.016](https://doi.org/10.1016/j.tourman.2009.02.016)

