import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the trained model
print("Loading trained model...")
with open('admission_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load college data
print("Loading college data...")
colleges_df = pd.read_csv('college_admissions.csv')

# Clean colleges (same as training)
colleges_clean = colleges_df[
    colleges_df['Percent admitted - total'].notna() &
    colleges_df['SAT Math 25th percentile score'].notna() &
    colleges_df['SAT Math 75th percentile score'].notna()
    ].copy()

print(f"Colleges available: {len(colleges_clean)}\n")


def recommend_ed_schools(student_profile, top_n=5, min_admission_prob=0.5, visualize=True):

    print("=" * 70)
    print("STUDENT PROFILE ANALYSIS")
    print("=" * 70)
    print(f"GPA: {student_profile['gpa']:.2f}")
    print(f"SAT Math: {student_profile['sat_math']}")
    print(f"SAT Reading: {student_profile['sat_reading']}")
    print(f"SAT Total: {student_profile['sat_math'] + student_profile['sat_reading']}")
    print(f"Research Experience: {'Yes' if student_profile['has_research'] else 'No'}")
    print(f"Leadership Score: {student_profile['leadership_score']}/5")
    print(f"AP Courses: {student_profile.get('num_ap_courses', 0)}")

    # Filter by preferences if provided
    colleges_filtered = colleges_clean.copy()

    if 'preferred_states' in student_profile and student_profile['preferred_states']:
        colleges_filtered = colleges_filtered[
            colleges_filtered['State abbreviation'].isin(student_profile['preferred_states'])
        ]
        print(f"\nFiltering by states: {student_profile['preferred_states']}")

    if 'preferred_regions' in student_profile and student_profile['preferred_regions']:
        colleges_filtered = colleges_filtered[
            colleges_filtered['Geographic region'].isin(student_profile['preferred_regions'])
        ]
        print(f"Filtering by regions: {student_profile['preferred_regions']}")

    print(f"\nEvaluating {len(colleges_filtered)} colleges...\n")

    # Prepare features for each college
    predictions = []

    for _, college in colleges_filtered.iterrows():
        # Create features (same as training)
        sat_total = student_profile['sat_math'] + student_profile['sat_reading']
        college_admit_rate = college['Percent admitted - total'] / 100
        college_sat_25 = college['SAT Math 25th percentile score']
        college_sat_75 = college['SAT Math 75th percentile score']

        sat_above_75 = 1 if student_profile['sat_math'] >= college_sat_75 else 0
        sat_above_25 = 1 if student_profile['sat_math'] >= college_sat_25 else 0
        gpa_high = 1 if student_profile['gpa'] >= 3.7 else 0
        sat_diff = student_profile['sat_math'] - college_sat_25

        # Calculate percentile position
        if college_sat_75 - college_sat_25 > 0:
            sat_percentile_position = sat_diff / (college_sat_75 - college_sat_25)
        else:
            sat_percentile_position = 0.5

        academic_strength = student_profile['gpa'] * sat_total / 1000

        features = [[
            student_profile['gpa'],
            student_profile['sat_math'],
            student_profile['sat_reading'],
            sat_total,
            student_profile['has_research'],
            student_profile['leadership_score'],
            student_profile.get('num_ap_courses', 0),
            college_admit_rate,
            sat_above_75,
            sat_above_25,
            gpa_high,
            sat_diff,
            sat_percentile_position,
            academic_strength
        ]]

        # Predict admission probability
        admit_prob = model.predict_proba(features)[0][1]  # Probability of admission

        predictions.append({
            'college_name': college['Name'],
            'state': college['State abbreviation'],
            'region': college['Geographic region'],
            'admit_probability': admit_prob,
            'college_admit_rate': college_admit_rate,
            'college_sat_25': college_sat_25,
            'college_sat_75': college_sat_75,
            'student_sat_vs_25': sat_diff,
            'student_sat_vs_75': student_profile['sat_math'] - college_sat_75
        })

    # Create DataFrame and sort by admission probability
    results_df = pd.DataFrame(predictions)
    results_df = results_df[results_df['admit_probability'] >= min_admission_prob]
    results_df = results_df.sort_values('admit_probability', ascending=False)

    # Add ED recommendation confidence
    results_df['ed_confidence'] = results_df['admit_probability'].apply(
        lambda x: 'HIGH' if x >= 0.75 else ('MEDIUM' if x >= 0.60 else 'LOW')
    )

    # Display results
    print("=" * 70)
    print(f"TOP {top_n} ED RECOMMENDATIONS")
    print("=" * 70)

    top_results = results_df.head(top_n)

    for idx, (i, row) in enumerate(top_results.iterrows(), 1):
        print(f"\n{idx}. {row['college_name']}")
        print(f"   Location: {row['state']} ({row['region']})")
        print(f"   Admission Probability: {row['admit_probability']:.1%}")
        print(f"   ED Confidence: {row['ed_confidence']}")
        print(f"   College Admit Rate: {row['college_admit_rate']:.1%}")
        print(f"   Your SAT vs College 25th: +{row['student_sat_vs_25']:.0f} points")
        print(f"   Your SAT vs College 75th: {row['student_sat_vs_75']:+.0f} points")

    print("\n" + "=" * 70)
    print("ED RECOMMENDATION GUIDANCE")
    print("=" * 70)

    if len(top_results) > 0:
        top_prob = top_results.iloc[0]['admit_probability']

        if top_prob >= 0.75:
            print("\✅ STRONG ED CANDIDATES")
            print("   You have excellent chances at these schools.")
            print("   Consider applying ED to your top choice from this list.")
        elif top_prob >= 0.60:
            print("\n️  MODERATE ED CANDIDATES")
            print("   Reasonable chances, but not guaranteed.")
            print("   Consider if this is truly your #1 choice before committing to ED.")
        else:
            print("\n REACH SCHOOLS")
            print("   Lower admission probabilities.")
            print("   Consider applying Regular Decision instead to keep options open.")
    else:
        print("\nNo schools meet the minimum admission probability threshold.")
        print("Consider broadening your search or applying Regular Decision.")

    # Create visualizations
    if visualize and len(top_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Admission probability by school
        ax1 = axes[0]
        colors = ['green' if x >= 0.75 else 'orange' if x >= 0.60 else 'red'
                  for x in top_results['admit_probability']]
        ax1.barh(range(len(top_results)), top_results['admit_probability'], color=colors)
        ax1.set_yticks(range(len(top_results)))
        ax1.set_yticklabels([name[:30] for name in top_results['college_name']], fontsize=9)
        ax1.set_xlabel('Admission Probability')
        ax1.set_title('ED Recommendation Probabilities')
        ax1.set_xlim(0, 1)
        ax1.axvline(x=0.75, color='green', linestyle='--', alpha=0.5, label='High (75%)')
        ax1.axvline(x=0.60, color='orange', linestyle='--', alpha=0.5, label='Medium (60%)')
        ax1.legend()

        # Plot 2: Your SAT vs College ranges
        ax2 = axes[1]
        college_names = [name[:20] for name in top_results['college_name'].head(5)]
        your_sat = [student_profile['sat_math']] * len(college_names)
        college_25 = top_results['college_sat_25'].head(5).values
        college_75 = top_results['college_sat_75'].head(5).values

        x = np.arange(len(college_names))
        width = 0.25

        ax2.bar(x - width, college_25, width, label='25th Percentile', alpha=0.7)
        ax2.bar(x, your_sat, width, label='Your SAT Math', alpha=0.7, color='green')
        ax2.bar(x + width, college_75, width, label='75th Percentile', alpha=0.7)

        ax2.set_ylabel('SAT Math Score')
        ax2.set_title('Your SAT vs College Ranges (Top 5)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(college_names, rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.set_ylim(300, 800)

        plt.tight_layout()
        plt.savefig('ed_recommendations.png', dpi=300, bbox_inches='tight')
        print("\n📊 Saved visualization: 'ed_recommendations.png'")
        plt.close()

    return results_df.head(top_n)


# Example usage
if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# COLLEGE COUNSELOR AI - ED RECOMMENDATION SYSTEM")
    print("#" * 70 + "\n")

    # Example student profile
    example_student = {
        'gpa': 3.8,
        'sat_math': 720,
        'sat_reading': 690,
        'has_research': 1,
        'leadership_score': 4,
        'num_ap_courses': 8,
        # Optional filters:
        # 'preferred_states': ['NC', 'VA', 'MA'],
        # 'preferred_regions': ['Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)']
    }

    recommendations = recommend_ed_schools(example_student, top_n=10, min_admission_prob=0.6, visualize=True)

    print("\n" + "=" * 70)
    print("\nTo use with your own profile, modify the 'example_student' dictionary!")
    print("\nExample with location filters:")
    print("  'preferred_states': ['NC', 'VA', 'MA']")
    print("  'preferred_regions': ['Southeast (...)']")
    print("\nVisualization saved as 'ed_recommendations.png'")