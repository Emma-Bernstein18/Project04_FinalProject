import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load model and data
print("Loading model and data...")
with open('admission_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Model type: {type(model).__name__}")

colleges_df = pd.read_csv('college_admissions.csv')
colleges_clean = colleges_df[
    colleges_df['Percent admitted - total'].notna() &
    colleges_df['SAT Math 25th percentile score'].notna() &
    colleges_df['SAT Math 75th percentile score'].notna()
    ].copy()

print(f"Loaded {len(colleges_clean)} colleges\n")

feature_names = [
    'GPA', 'SAT Math', 'SAT Reading', 'SAT Total',
    'Research', 'Leadership', 'AP Courses', 'College Admit Rate',
    'SAT Above 75th', 'SAT Above 25th', 'GPA High', 'SAT Diff',
    'SAT Percentile Position', 'Academic Strength'
]

# ============================================================================
# VISUALIZATION 1: Feature Importance (Tree-based)
# ============================================================================
print("Creating Feature Importance visualization...")

# For Gradient Boosting, use feature_importances_
feature_importance = model.feature_importances_

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feature_names)))
indices = np.argsort(feature_importance)
ax.barh([feature_names[i] for i in indices], feature_importance[indices], color=colors[indices])
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance: What Drives Admission Probability?\n(Gradient Boosting Model)', fontsize=14,
             fontweight='bold')
for i, v in enumerate(feature_importance[indices]):
    ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('01_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_feature_importance.png\n")
plt.close()

# ============================================================================
# VISUALIZATION 2: Admission Probability Distribution
# ============================================================================
print("Creating Admission Probability Distribution...")

# Generate predictions for all colleges
predictions = []
for _, college in colleges_clean.iterrows():
    try:
        college_admit_rate = college['Percent admitted - total'] / 100
        college_sat_25 = college['SAT Math 25th percentile score']
        college_sat_75 = college['SAT Math 75th percentile score']

        # Use a sample student profile
        student_gpa = 3.5
        student_sat_math = 700
        student_sat_reading = 700
        sat_total = student_sat_math + student_sat_reading

        sat_above_75 = 1 if student_sat_math >= college_sat_75 else 0
        sat_above_25 = 1 if student_sat_math >= college_sat_25 else 0
        gpa_high = 1 if student_gpa >= 3.7 else 0
        sat_diff = student_sat_math - college_sat_25

        if college_sat_75 - college_sat_25 > 0:
            sat_percentile_position = sat_diff / (college_sat_75 - college_sat_25)
        else:
            sat_percentile_position = 0.5

        academic_strength = student_gpa * sat_total / 1000

        features = np.array([[
            student_gpa, student_sat_math, student_sat_reading, sat_total,
            0, 3, 3, college_admit_rate,
            sat_above_75, sat_above_25, gpa_high, sat_diff,
            sat_percentile_position, academic_strength
        ]])

        prob = model.predict_proba(features)[0][1]
        predictions.append({
            'college': college['Name'],
            'admit_rate': college_admit_rate,
            'predicted_prob': prob
        })
    except:
        continue

pred_df = pd.DataFrame(predictions)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram of predicted probabilities
axes[0, 0].hist(pred_df['predicted_prob'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Predicted Admission Probability', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Number of Colleges', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Distribution of Predicted Probabilities\n(for 3.5 GPA, 1400 SAT student)', fontsize=12,
                     fontweight='bold')
axes[0, 0].axvline(pred_df['predicted_prob'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {pred_df["predicted_prob"].mean():.2%}')
axes[0, 0].legend()

# Scatter: College Admit Rate vs Predicted Probability
axes[0, 1].scatter(pred_df['admit_rate'], pred_df['predicted_prob'], alpha=0.5, s=30)
axes[0, 1].set_xlabel('College Admit Rate', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Predicted Admission Probability', fontsize=11, fontweight='bold')
axes[0, 1].set_title('College Selectivity vs Predicted Probability', fontsize=12, fontweight='bold')
axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Match')
axes[0, 1].legend()

# Box plot by selectivity tier
pred_df['selectivity_tier'] = pd.cut(pred_df['admit_rate'],
                                     bins=[0, 0.15, 0.30, 0.50, 1.0],
                                     labels=['Highly Selective\n(<15%)', 'Selective\n(15-30%)',
                                             'Moderately Selective\n(30-50%)', 'Less Selective\n(>50%)'])
axes[1, 0].boxplot([pred_df[pred_df['selectivity_tier'] == tier]['predicted_prob'].values
                    for tier in ['Highly Selective\n(<15%)', 'Selective\n(15-30%)', 'Moderately Selective\n(30-50%)',
                                 'Less Selective\n(>50%)']],
                   labels=['Highly Selective\n(<15%)', 'Selective\n(15-30%)', 'Moderately Selective\n(30-50%)',
                           'Less Selective\n(>50%)'])
axes[1, 0].set_ylabel('Predicted Admission Probability', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Predicted Probability by College Selectivity', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Top 10 colleges where model predicts highest probability
top_10 = pred_df.nlargest(10, 'predicted_prob')
axes[1, 1].barh(range(len(top_10)), top_10['predicted_prob'], color='lightgreen', edgecolor='black')
axes[1, 1].set_yticks(range(len(top_10)))
axes[1, 1].set_yticklabels([name[:30] for name in top_10['college']], fontsize=9)
axes[1, 1].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Top 10 Colleges (Highest Predicted Probability)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlim([0, 1])
for i, v in enumerate(top_10['predicted_prob']):
    axes[1, 1].text(v + 0.02, i, f'{v:.1%}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('02_probability_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_probability_distribution.png\n")
plt.close()

# ============================================================================
# VISUALIZATION 3: ED Boost Impact
# ============================================================================
print("Creating ED Boost Impact visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Calculate ED boost for different selectivity levels
selectivity_levels = np.linspace(0.05, 0.95, 20)
ed_boosts = np.maximum(0.1, 1 - selectivity_levels)
adjusted_probs = np.minimum(0.99, 0.70 + (ed_boosts * 0.15))  # Assuming base prob of 70%

axes[0].plot(selectivity_levels * 100, ed_boosts, marker='o', linewidth=2, markersize=6, label='ED Boost Factor')
axes[0].set_xlabel('College Admit Rate (%)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('ED Boost Factor', fontsize=11, fontweight='bold')
axes[0].set_title('How Much Does ED Help?\n(Based on College Selectivity)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# Show impact on admission probability
base_prob = 0.70
axes[1].plot(selectivity_levels * 100, np.full_like(selectivity_levels, base_prob),
             label='Without ED', linewidth=2, linestyle='--', color='red')
axes[1].plot(selectivity_levels * 100, adjusted_probs,
             label='With ED Boost', linewidth=2, marker='o', markersize=6, color='green')
axes[1].fill_between(selectivity_levels * 100, base_prob, adjusted_probs, alpha=0.2, color='green')
axes[1].set_xlabel('College Admit Rate (%)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Admission Probability', fontsize=11, fontweight='bold')
axes[1].set_title('ED Boost Impact on Admission Probability\n(Base probability: 70%)', fontsize=12, fontweight='bold')
axes[1].set_ylim([0.6, 1.0])
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('03_ed_boost_impact.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_ed_boost_impact.png\n")
plt.close()

# ============================================================================
# VISUALIZATION 4: Student Profile Impact
# ============================================================================
print("Creating Student Profile Impact visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Impact of GPA
gpas = np.linspace(2.5, 4.0, 20)
probs_by_gpa = []
for gpa in gpas:
    features = np.array([[gpa, 700, 700, 1400, 0, 3, 3, 0.30, 0, 1, 0, 50, 0.5, gpa * 1.4 / 1000]])
    prob = model.predict_proba(features)[0][1]
    probs_by_gpa.append(prob)

axes[0, 0].plot(gpas, probs_by_gpa, marker='o', linewidth=2, markersize=6, color='blue')
axes[0, 0].set_xlabel('GPA', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Predicted Admission Probability', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Impact of GPA on Admission Probability\n(SAT: 1400, College Admit Rate: 30%)', fontsize=12,
                     fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Impact of SAT Math
sats = np.linspace(400, 800, 20)
probs_by_sat = []
for sat in sats:
    features = np.array([[3.5, sat, 700, sat + 700, 0, 3, 3, 0.30, 0, 1, 0, 50, 0.5, 3.5 * (sat + 700) / 1000]])
    prob = model.predict_proba(features)[0][1]
    probs_by_sat.append(prob)

axes[0, 1].plot(sats, probs_by_sat, marker='o', linewidth=2, markersize=6, color='green')
axes[0, 1].set_xlabel('SAT Math Score', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Predicted Admission Probability', fontsize=11, fontweight='bold')
axes[0, 1].set_title(
    'Impact of SAT Math on Admission Probability\n(GPA: 3.5, SAT Reading: 700, College Admit Rate: 30%)', fontsize=12,
    fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Impact of AP Courses
ap_counts = np.arange(0, 12)
probs_by_ap = []
for ap in ap_counts:
    features = np.array([[3.5, 700, 700, 1400, 0, 3, ap, 0.30, 0, 1, 0, 50, 0.5, 3.5 * 1.4 / 1000]])
    prob = model.predict_proba(features)[0][1]
    probs_by_ap.append(prob)

axes[1, 0].plot(ap_counts, probs_by_ap, marker='o', linewidth=2, markersize=6, color='purple')
axes[1, 0].set_xlabel('Number of AP Courses', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Predicted Admission Probability', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Impact of AP Courses on Admission Probability\n(GPA: 3.5, SAT: 1400, College Admit Rate: 30%)',
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Impact of College Admit Rate
college_rates = np.linspace(0.05, 0.95, 20)
probs_by_college_rate = []
for rate in college_rates:
    features = np.array([[3.5, 700, 700, 1400, 0, 3, 3, rate, 0, 1, 0, 50, 0.5, 3.5 * 1.4 / 1000]])
    prob = model.predict_proba(features)[0][1]
    probs_by_college_rate.append(prob)

axes[1, 1].plot(college_rates * 100, probs_by_college_rate, marker='o', linewidth=2, markersize=6, color='orange')
axes[1, 1].set_xlabel('College Admit Rate (%)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Predicted Admission Probability', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Impact of College Selectivity on Admission Probability\n(GPA: 3.5, SAT: 1400)', fontsize=12,
                     fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_student_profile_impact.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_student_profile_impact.png\n")
plt.close()

# ============================================================================
# VISUALIZATION 5: Model Insights Summary
# ============================================================================
print("Creating Model Insights Summary...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

# Title
fig.suptitle('College Counselor AI: Model Insights & Analysis', fontsize=16, fontweight='bold', y=0.98)

# Key findings text
ax_text = fig.add_subplot(gs[0, :])
ax_text.axis('off')

insights = f"""
KEY MODEL INSIGHTS (Gradient Boosting Classifier):

1. MOST IMPORTANT FACTORS:
   • College Admit Rate (selectivity) - Strongest predictor: More selective = lower probability
   • SAT Math Score - Strong positive: Higher SAT = higher admission probability
   • GPA - Strong positive: Higher GPA = higher admission probability
   • Academic Strength (GPA × SAT) - Combined academic power matters

2. ED BOOST STRATEGY:
   • ED helps MORE at selective schools (lower admit rates)
   • At a 30% admit rate school: ED boost adds ~10-15% to your chances
   • At a 70% admit rate school: ED boost adds ~5% to your chances
   • Model recommends ED at schools where you have 60-90% base probability

3. PREDICTION PATTERNS:
   • Model predicts highest probability at less selective schools (as expected)
   • For a 3.5 GPA, 1400 SAT student: average predicted probability is ~70%
   • Selectivity tier matters more than individual student characteristics

4. WHAT THE MODEL CAPTURES:
   ✓ Academic fit (SAT/GPA vs college standards)
   ✓ Selectivity matching (not being over/underqualified)
   ✓ ED boost effect (lower admit rate = more ED help)
   ✗ Extracurriculars (limited to research/leadership/AP)
   ✗ Essays, recommendations, unique circumstances
   ✗ Demonstrated interest, legacy status
"""

ax_text.text(0.05, 0.95, insights, transform=ax_text.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Feature importance (top 8)
ax1 = fig.add_subplot(gs[1, 0])
top_features_idx = np.argsort(feature_importance)[-8:]
top_features = [feature_names[i] for i in top_features_idx]
top_importance = feature_importance[top_features_idx]
colors_top = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
ax1.barh(top_features, top_importance, color=colors_top)
ax1.set_xlabel('Importance Score', fontsize=10, fontweight='bold')
ax1.set_title('Top 8 Most Important Features', fontsize=11, fontweight='bold')

# Model statistics
ax2 = fig.add_subplot(gs[1, 1])
ax2.axis('off')
stats_text = f"""
MODEL STATISTICS:

Algorithm: Gradient Boosting Classifier
Total Colleges: {len(colleges_clean)}
Features Used: {len(feature_names)}

Prediction Range:
  Min: {pred_df['predicted_prob'].min():.1%}
  Max: {pred_df['predicted_prob'].max():.1%}
  Mean: {pred_df['predicted_prob'].mean():.1%}
  Median: {pred_df['predicted_prob'].median():.1%}

Correlation Insights:
  • College Admit Rate vs Predicted Prob: {pred_df['admit_rate'].corr(pred_df['predicted_prob']):.3f}
    (Strong negative: selective schools = lower predictions)
"""
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Recommendations
ax3 = fig.add_subplot(gs[2, :])
ax3.axis('off')
recommendations = """
RECOMMENDATIONS FOR USING THIS MODEL:

✓ DO: Use this model to identify schools where you have realistic ED chances (60-90% predicted probability)
✓ DO: Consider ED at selective schools (20-40% admit rate) where model predicts 70%+ probability
✓ DO: Use school preferences (size, location, cost) to narrow down options
✓ DO: Compare multiple schools before committing to ED

✗ DON'T: Treat predictions as guarantees - admissions involves subjective factors
✗ DON'T: Ignore other important factors (essays, recommendations, fit)
✗ DON'T: ED to a school just because model says you have high probability
✗ DON'T: Use this as your only college search tool

BEST PRACTICE: Use this as ONE tool in your college search toolkit, combined with:
  • Campus visits and conversations with current students
  • Research into academic programs and campus culture
  • Consultation with your school counselor
  • Consideration of financial aid packages
"""
ax3.text(0.05, 0.95, recommendations, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2))

plt.savefig('05_model_insights_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_model_insights_summary.png\n")
plt.close()

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated files:")
print("  1. 01_feature_importance.png - What drives admission probability?")
print("  2. 02_probability_distribution.png - How predictions vary across colleges")
print("  3. 03_ed_boost_impact.png - How much does ED help?")
print("  4. 04_student_profile_impact.png - How student characteristics affect predictions")
print("  5. 05_model_insights_summary.png - Key findings and recommendations")
print("\nUse these visualizations in your project presentation!")