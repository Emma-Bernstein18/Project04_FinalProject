# **College Counselor AI: ED School Recommendation System**

**A machine learning system that predicts ED admission probability and recommends best-fit schools based on student profiles and preferences.**

## **What This Project Does**

College Counselor AI helps high school students make informed Early Decision (ED) college choices by:

1. **Predicting admission probability** using a trained ML model on 1,182 colleges
2. **Calculating ED boost** based on college selectivity
3. **Ranking schools** by ED Fit Score (combines base + ED probability)
4. **Filtering by preferences** (school type, size, location, cost, financial aid)
5. **Providing strategic guidance** on when ED is worth it

## **Key Features**

### **Student Profile Input**
- GPA (0.0 - 4.0)
- SAT Math & Reading (200 - 800 each)
- AP Courses (0 - 15)

### **Preference Filters**
- School Type: Public, Private (multi-select)
- School Size: Small, Medium, Large
- Location: Urban, Suburban, Rural
- HBCU Status: Include/Exclude
- Max Cost: $0 - $80,000
- Min Financial Aid %: 0% - 100%

### **Recommendation Output**
- Top 10 schools ranked by ED Fit Score
- School characteristics (type, size, location, cost, financial aid)
- Admission probabilities (with and without ED)
- College admit rate and selectivity tier



### **Source:** College Scorecard Dataset (U.S. Department of Education)
- **Colleges:** 1,182 four-year institutions
- **Features:** 14 per college
- **Time Period:** 2013-14 academic year
- **Coverage:** Public, private, and HBCU institutions

**Data Cleaning:**
- Removed colleges with missing critical data
- Standardized numerical values


## **Model: Gradient Boosting Classifier**

**Why Gradient Boosting?**
- Captures non-linear relationships
- Handles feature interactions automatically


## **Key Findings**

### **Finding #1: SAT Percentile Position Dominates (66.7%)**
Where you rank within a college's applicant pool matters 5x more than college selectivity. Colleges compare you to their typical students, not to a universal standard.

### **Finding #2: College Selectivity (13.6%)**
Choosing the right selectivity tier matters more than optimizing your profile.

### **Finding #3: ED is Strategic**
ED helps more at selective schools.



## **Limitations**

### **Data Limitations**
- Dataset is from 2013-14 (admissions have evolved)
- Limited to 1,182 colleges
- No international colleges

### **Model Limitations**
- Cannot capture essays and personal statements
- Cannot measure teacher recommendations
- Cannot assess demonstrated interest
- Cannot account for unique circumstances
- Cannot measure legacy status
- Cannot assess extracurricular depth
- Cannot evaluate holistic review factors

### **Important Disclaimer**
 **This model is a tool to help you think strategically about college search, not a crystal ball. Real admissions are more complex than numbers alone. Use this as ONE tool combined with campus visits, counselor advice, and your own research.**

---

## **Installation**

### **Requirements**
- Python 3.7+
- pandas, numpy, scikit-learn
- matplotlib, seaborn

### **Setup**
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the GUI
python gui.py
```

---
### **Step-by-Step**
1. Enter your profile (GPA, SAT, etc.)
2. Set preferences (school type, size, etc.)
3. Click "Get Recommendations"
4. Review top 10 schools ranked by ED Fit Score
5. Analyze admission probabilities and school details

# **Thank you!**
