import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Loading college data...")
colleges_df = pd.read_csv('college_admissions.csv')

# Clean the data - keep only colleges with complete admissions data
print("\nCleaning data...")
colleges_clean = colleges_df[
    colleges_df['Percent admitted - total'].notna() &
    colleges_df['SAT Math 25th percentile score'].notna() &
    colleges_df['SAT Math 75th percentile score'].notna()
    ].copy()

print(f"Colleges with complete data: {len(colleges_clean)}")

# Generate synthetic students
print("\nGenerating synthetic student applications...")

student_applications = []
student_id = 1
num_students = 2000  # Total unique students
apps_per_student = 8  # Each student applies to 8 colleges

for i in range(num_students):
    # Generate a student profile with realistic distributions
    student_gpa = np.random.normal(3.5, 0.4)  # Mean 3.5, SD 0.4
    student_gpa = np.clip(student_gpa, 2.0, 4.0)  # Keep between 2.0-4.0

    student_sat_math = np.random.normal(600, 100)
    student_sat_math = np.clip(student_sat_math, 400, 800)

    student_sat_reading = np.random.normal(590, 100)
    student_sat_reading = np.clip(student_sat_reading, 400, 800)

    student_sat_total = student_sat_math + student_sat_reading

    # Random demographics and activities
    has_research = np.random.choice([0, 1], p=[0.7, 0.3])
    leadership_score = np.random.randint(1, 6)  # 1-5 scale
    num_ap_courses = np.random.randint(0, 12)  # 0-11 AP courses

    # Student applies to random colleges
    selected_colleges = colleges_clean.sample(n=apps_per_student)

    for _, college in selected_colleges.iterrows():
        # Calculate admission probability based on fit
        college_sat_25 = college['SAT Math 25th percentile score']
        college_sat_75 = college['SAT Math 75th percentile score']
        college_admit_rate = college['Percent admitted - total'] / 100

        # Admission probability model based on student vs college stats
        if student_sat_math >= college_sat_75:
            base_prob = min(0.85, college_admit_rate * 2.5)
        elif student_sat_math >= college_sat_25:
            base_prob = college_admit_rate * 1.5
        else:
            base_prob = college_admit_rate * 0.5

        # Adjust for GPA and other factors
        if student_gpa >= 3.7:
            base_prob *= 1.2
        if has_research:
            base_prob *= 1.1
        if leadership_score >= 4:
            base_prob *= 1.1
        if num_ap_courses >= 8:
            base_prob *= 1.15

        base_prob = min(base_prob, 0.95)  # Cap at 95%

        # Determine admission outcome
        admitted = 1 if np.random.random() < base_prob else 0

        # Add application record
        student_applications.append({
            'student_id': student_id,
            'college_name': college['Name'],
            'student_gpa': round(student_gpa, 2),
            'student_sat_math': int(student_sat_math),
            'student_sat_reading': int(student_sat_reading),
            'student_sat_total': int(student_sat_total),
            'has_research_experience': has_research,
            'leadership_score': leadership_score,
            'num_ap_courses': num_ap_courses,
            'college_admit_rate': college_admit_rate,
            'college_sat_math_25': college_sat_25,
            'college_sat_math_75': college_sat_75,
            'state': college['State abbreviation'],
            'region': college['Geographic region'],
            'admitted': admitted
        })

    student_id += 1

    if (i + 1) % 500 == 0:
        print(f"Generated {i + 1} students...")

# Create DataFrame
student_df = pd.DataFrame(student_applications)

print(f"\nTotal applications generated: {len(student_df)}")
print(f"Admission rate in synthetic data: {student_df['admitted'].mean():.2%}")
print(f"\nSample of generated data:")
print(student_df.head(10))

# Save to CSV
student_df.to_csv('student_applications.csv', index=False)
print("\n✅ Saved to 'student_applications.csv'")
print("\nYou now have your training data ready!")