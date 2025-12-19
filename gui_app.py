import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import pickle
import traceback
import warnings

warnings.filterwarnings('ignore')

# State abbreviation mapping
STATE_ABBREV_TO_NAME = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'DC': 'District of Columbia', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii',
    'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine',
    'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska',
    'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
    'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island',
    'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas',
    'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

# Load model and data
print("Loading model...")
try:
    with open('admission_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

print("Loading college data...")
try:
    colleges_df = pd.read_csv('college_admissions.csv')
    colleges_clean = colleges_df[
        colleges_df['Percent admitted - total'].notna() &
        colleges_df['SAT Math 25th percentile score'].notna() &
        colleges_df['SAT Math 75th percentile score'].notna()
        ].copy()
    print(f"✓ College data loaded ({len(colleges_clean)} colleges)\n")

except Exception as e:
    print(f"✗ Error loading college data: {e}")
    colleges_clean = None


class CollegeCounselorAI:
    def __init__(self, root):
        self.root = root
        self.root.title("College Counselor AI - ED Recommendation System")
        self.root.geometry("1000x900")
        self.root.configure(bg='#f0f0f0')

        # Create main frame with scrollbar
        main_canvas = tk.Canvas(root, bg='#f0f0f0', highlightthickness=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        main_canvas.pack(side="left", fill="both", expand=True, padx=15, pady=15)
        scrollbar.pack(side="right", fill="y")

        # Title
        title = ttk.Label(scrollable_frame, text="College Counselor AI", font=('Arial', 18, 'bold'))
        title.pack(pady=10)

        subtitle = ttk.Label(scrollable_frame, text="Enter Your Profile to Get ED Recommendations", font=('Arial', 11))
        subtitle.pack(pady=5)

        # ACADEMIC PROFILE
        academic_frame = ttk.LabelFrame(scrollable_frame, text="Academic Profile", padding=15)
        academic_frame.pack(fill='x', pady=10, padx=5)

        ttk.Label(academic_frame, text="GPA:", font=('Arial', 10)).grid(row=0, column=0, sticky='w', pady=8, padx=5)
        self.gpa_var = tk.StringVar(value="3.94")
        ttk.Entry(academic_frame, textvariable=self.gpa_var, width=8).grid(row=0, column=1, sticky='w', padx=5)

        ttk.Label(academic_frame, text="SAT Math:", font=('Arial', 10)).grid(row=0, column=2, sticky='w', padx=5)
        self.sat_math_var = tk.StringVar(value="790")
        ttk.Entry(academic_frame, textvariable=self.sat_math_var, width=8).grid(row=0, column=3, sticky='w', padx=5)

        ttk.Label(academic_frame, text="SAT Reading:", font=('Arial', 10)).grid(row=0, column=4, sticky='w', padx=5)
        self.sat_reading_var = tk.StringVar(value="740")
        ttk.Entry(academic_frame, textvariable=self.sat_reading_var, width=8).grid(row=0, column=5, sticky='w', padx=5)

        ttk.Label(academic_frame, text="AP Courses:", font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=8,
                                                                               padx=5)
        self.ap_var = tk.StringVar(value="3")
        ttk.Entry(academic_frame, textvariable=self.ap_var, width=8).grid(row=1, column=1, sticky='w', padx=5)

        # SCHOOL PREFERENCES
        pref_frame = ttk.LabelFrame(scrollable_frame, text="School Preferences", padding=15)
        pref_frame.pack(fill='x', pady=10, padx=5)

        # States
        ttk.Label(pref_frame, text="Preferred States (e.g., NC,VA,MA):", font=('Arial', 10)).grid(row=0, column=0,
                                                                                                  columnspan=2,
                                                                                                  sticky='w', pady=8,
                                                                                                  padx=5)
        self.states_var = tk.StringVar(value="NC,VA,MA")
        ttk.Entry(pref_frame, textvariable=self.states_var, width=40).grid(row=0, column=2, columnspan=3, sticky='ew',
                                                                           padx=5)

        # Public vs Private
        ttk.Label(pref_frame, text="School Type:", font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=8, padx=5)
        self.school_type_var = tk.StringVar(value="Any")
        school_type_combo = ttk.Combobox(pref_frame, textvariable=self.school_type_var,
                                         values=["Any", "Public", "Private"], state="readonly", width=15)
        school_type_combo.grid(row=1, column=1, sticky='w', padx=5)

        # School Size
        ttk.Label(pref_frame, text="School Size:", font=('Arial', 10)).grid(row=1, column=2, sticky='w', padx=5)
        self.size_var = tk.StringVar(value="Any")
        size_combo = ttk.Combobox(pref_frame, textvariable=self.size_var,
                                  values=["Any", "Small (<5k)", "Medium (5k-15k)", "Large (>15k)"], state="readonly",
                                  width=15)
        size_combo.grid(row=1, column=3, sticky='w', padx=5)

        # Urban/Rural
        ttk.Label(pref_frame, text="Location Type:", font=('Arial', 10)).grid(row=2, column=0, sticky='w', pady=8,
                                                                              padx=5)
        self.location_var = tk.StringVar(value="Any")
        location_combo = ttk.Combobox(pref_frame, textvariable=self.location_var,
                                      values=["Any", "Urban", "Suburban", "Rural"], state="readonly", width=15)
        location_combo.grid(row=2, column=1, sticky='w', padx=5)

        # HBCU
        ttk.Label(pref_frame, text="HBCU:", font=('Arial', 10)).grid(row=2, column=2, sticky='w', padx=5)
        self.hbcu_var = tk.StringVar(value="Any")
        hbcu_combo = ttk.Combobox(pref_frame, textvariable=self.hbcu_var,
                                  values=["Any", "Yes", "No"], state="readonly", width=15)
        hbcu_combo.grid(row=2, column=3, sticky='w', padx=5)

        # Max Cost
        ttk.Label(pref_frame, text="Max Annual Cost ($):", font=('Arial', 10)).grid(row=3, column=0, sticky='w', pady=8,
                                                                                    padx=5)
        self.max_cost_var = tk.StringVar(value="")
        ttk.Entry(pref_frame, textvariable=self.max_cost_var, width=15).grid(row=3, column=1, sticky='w', padx=5)

        # Min Financial Aid
        ttk.Label(pref_frame, text="Min % Receiving Aid:", font=('Arial', 10)).grid(row=3, column=2, sticky='w', padx=5)
        self.min_aid_var = tk.StringVar(value="")
        ttk.Entry(pref_frame, textvariable=self.min_aid_var, width=15).grid(row=3, column=3, sticky='w', padx=5)

        pref_frame.columnconfigure(2, weight=1)
        pref_frame.columnconfigure(3, weight=1)

        # Button
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=15)
        ttk.Button(button_frame, text="Get ED Recommendations", command=self.get_recommendations).pack()

        # Results section
        results_label = ttk.Label(scrollable_frame, text="Recommendations", font=('Arial', 12, 'bold'))
        results_label.pack(anchor='w', pady=(10, 5), padx=5)

        self.results_text = scrolledtext.ScrolledText(scrollable_frame, wrap=tk.WORD, font=('Courier', 9), height=28)
        self.results_text.pack(fill='both', expand=True, padx=5)

    def get_recommendations(self):
        """Get ED recommendations based on user input"""
        try:
            print("\n--- Getting recommendations ---")

            if model is None:
                raise Exception("Model not loaded. Check if admission_model.pkl exists.")
            if colleges_clean is None:
                raise Exception("College data not loaded. Check if college_admissions.csv exists.")

            # Validate and get user input
            try:
                gpa = float(self.gpa_var.get())
                sat_math = int(self.sat_math_var.get())
                sat_reading = int(self.sat_reading_var.get())
                ap_courses = int(self.ap_var.get())
            except ValueError as e:
                raise Exception(f"Invalid input: {e}. Please enter valid numbers.")

            student_profile = {
                'gpa': gpa,
                'sat_math': sat_math,
                'sat_reading': sat_reading,
                'has_research': 0,  # Removed from UI
                'leadership_score': 3,  # Default value
                'num_ap_courses': ap_courses,
            }

            # Parse preferences
            states_input = self.states_var.get().strip()
            if states_input:
                abbrevs = [s.strip().upper() for s in states_input.split(',')]
                state_names = [STATE_ABBREV_TO_NAME.get(abbr, abbr) for abbr in abbrevs]
                student_profile['preferred_states'] = state_names

            student_profile['school_type'] = self.school_type_var.get()
            student_profile['size'] = self.size_var.get()
            student_profile['location'] = self.location_var.get()
            student_profile['hbcu'] = self.hbcu_var.get()

            if self.max_cost_var.get():
                student_profile['max_cost'] = float(self.max_cost_var.get())

            if self.min_aid_var.get():
                student_profile['min_aid'] = float(self.min_aid_var.get())

            print(f"Student profile: {student_profile}")

            # Get recommendations
            recommendations = self.recommend_ed_schools(student_profile)
            print(f"Got {len(recommendations)} recommendations")

            # Display results
            self.display_results(student_profile, recommendations)
            print("Results displayed")

        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)

    def recommend_ed_schools(self, student_profile, top_n=10):
        """Get ED recommendations with preference filtering"""
        colleges_filtered = colleges_clean.copy()

        # Filter by states
        if 'preferred_states' in student_profile:
            colleges_filtered = colleges_filtered[
                colleges_filtered['State abbreviation'].isin(student_profile['preferred_states'])
            ]
            print(f"After state filter: {len(colleges_filtered)} colleges")

        # Filter by school type (Public/Private)
        if student_profile['school_type'] != 'Any':
            colleges_filtered = colleges_filtered[
                colleges_filtered['Control of institution'] == student_profile['school_type']
                ]
            print(f"After school type filter: {len(colleges_filtered)} colleges")

        # Filter by school size
        if student_profile['size'] != 'Any':
            if student_profile['size'] == 'Small (<5k)':
                colleges_filtered = colleges_filtered[
                    colleges_filtered['Estimated undergraduate enrollment, total'] < 5000
                    ]
            elif student_profile['size'] == 'Medium (5k-15k)':
                colleges_filtered = colleges_filtered[
                    (colleges_filtered['Estimated undergraduate enrollment, total'] >= 5000) &
                    (colleges_filtered['Estimated undergraduate enrollment, total'] <= 15000)
                    ]
            elif student_profile['size'] == 'Large (>15k)':
                colleges_filtered = colleges_filtered[
                    colleges_filtered['Estimated undergraduate enrollment, total'] > 15000
                    ]
            print(f"After size filter: {len(colleges_filtered)} colleges")

        # Filter by location type
        if student_profile['location'] != 'Any':
            if student_profile['location'] == 'Urban':
                colleges_filtered = colleges_filtered[
                    colleges_filtered['Degree of urbanization (Urban-centric locale)'].str.contains('City', na=False)
                ]
            elif student_profile['location'] == 'Suburban':
                colleges_filtered = colleges_filtered[
                    colleges_filtered['Degree of urbanization (Urban-centric locale)'].str.contains('Suburb', na=False)
                ]
            elif student_profile['location'] == 'Rural':
                colleges_filtered = colleges_filtered[
                    colleges_filtered['Degree of urbanization (Urban-centric locale)'].str.contains('Rural|Town',
                                                                                                    na=False,
                                                                                                    regex=True)
                ]
            print(f"After location filter: {len(colleges_filtered)} colleges")

        # Filter by HBCU
        if student_profile['hbcu'] != 'Any':
            colleges_filtered = colleges_filtered[
                colleges_filtered['Historically Black College or University'] == student_profile['hbcu']
                ]
            print(f"After HBCU filter: {len(colleges_filtered)} colleges")

        # Filter by max cost
        if 'max_cost' in student_profile:
            colleges_filtered = colleges_filtered[
                colleges_filtered['Total price for out-of-state students living on campus 2013-14'] <= student_profile[
                    'max_cost']
                ]
            print(f"After cost filter: {len(colleges_filtered)} colleges")

        # Filter by min financial aid
        if 'min_aid' in student_profile:
            colleges_filtered = colleges_filtered[
                colleges_filtered['Percent of freshmen receiving any financial aid'] >= student_profile['min_aid']
                ]
            print(f"After financial aid filter: {len(colleges_filtered)} colleges")

        predictions = []
        count = 0

        for idx, (_, college) in enumerate(colleges_filtered.iterrows()):
            try:
                sat_total = student_profile['sat_math'] + student_profile['sat_reading']
                college_admit_rate = college['Percent admitted - total'] / 100
                college_sat_25 = college['SAT Math 25th percentile score']
                college_sat_75 = college['SAT Math 75th percentile score']

                sat_above_75 = 1 if student_profile['sat_math'] >= college_sat_75 else 0
                sat_above_25 = 1 if student_profile['sat_math'] >= college_sat_25 else 0
                gpa_high = 1 if student_profile['gpa'] >= 3.7 else 0
                sat_diff = student_profile['sat_math'] - college_sat_25

                if college_sat_75 - college_sat_25 > 0:
                    sat_percentile_position = sat_diff / (college_sat_75 - college_sat_25)
                else:
                    sat_percentile_position = 0.5

                academic_strength = student_profile['gpa'] * sat_total / 1000

                features = np.array([[
                    student_profile['gpa'],
                    student_profile['sat_math'],
                    student_profile['sat_reading'],
                    sat_total,
                    student_profile['has_research'],
                    student_profile['leadership_score'],
                    student_profile['num_ap_courses'],
                    college_admit_rate,
                    sat_above_75,
                    sat_above_25,
                    gpa_high,
                    sat_diff,
                    sat_percentile_position,
                    academic_strength
                ]])

                admit_prob = model.predict_proba(features)[0][1]

                selectivity_match = abs(admit_prob - college_admit_rate)
                ed_boost = max(0.1, 1 - college_admit_rate)
                adjusted_prob = min(0.99, admit_prob + (ed_boost * 0.15))

                ed_fit_score = (
                        (adjusted_prob * 0.4) +
                        ((1 - college_admit_rate) * 0.4) +
                        ((1 - min(selectivity_match, 0.5)) * 0.2)
                )

                if idx < 5:
                    print(f"  {college['Name']}: fit_score={ed_fit_score:.3f}")

                predictions.append({
                    'college_name': college['Name'],
                    'state': college['State abbreviation'],
                    'admit_probability': admit_prob,
                    'adjusted_probability': adjusted_prob,
                    'college_admit_rate': college_admit_rate,
                    'college_sat_25': college_sat_25,
                    'student_sat_vs_25': sat_diff,
                    'ed_fit_score': ed_fit_score,
                    'school_type': college['Control of institution'],
                    'size': college['Estimated undergraduate enrollment, total'],
                    'location': college['Degree of urbanization (Urban-centric locale)'],
                    'cost': college['Total price for out-of-state students living on campus 2013-14'],
                    'financial_aid': college['Percent of freshmen receiving any financial aid']
                })
                count += 1
            except Exception as e:
                continue

        print(f"Processed {count} colleges successfully")

        if len(predictions) == 0:
            print("No predictions made!")
            return pd.DataFrame()

        results_df = pd.DataFrame(predictions)
        results_df = results_df.sort_values('ed_fit_score', ascending=False)

        results_df['ed_confidence'] = results_df['adjusted_probability'].apply(
            lambda x: 'HIGH' if x >= 0.75 else ('MEDIUM' if x >= 0.60 else 'LOW')
        )

        return results_df.head(top_n)

    def display_results(self, student_profile, recommendations):
        """Display results"""
        self.results_text.delete(1.0, tk.END)

        output = "=" * 95 + "\n"
        output += "COLLEGE COUNSELOR AI - ED RECOMMENDATIONS\n"
        output += "=" * 95 + "\n\n"

        output += "YOUR PROFILE\n"
        output += "-" * 95 + "\n"
        output += f"GPA: {student_profile['gpa']:.2f}  |  SAT: {student_profile['sat_math'] + student_profile['sat_reading']}  |  AP Courses: {student_profile['num_ap_courses']}\n\n"

        output += "TOP ED RECOMMENDATIONS (Ranked by ED Fit)\n"
        output += "-" * 95 + "\n\n"

        if len(recommendations) == 0:
            output += "No schools match your preferences. Try adjusting your filters.\n"
        else:
            for idx, (i, row) in enumerate(recommendations.iterrows(), 1):
                output += f"{idx}. {row['college_name']}\n"
                output += f"   Type: {row['school_type']}  |  Size: {row['size']:.0f} students  |  Location: {row['location']}\n"
                output += f"   Cost: ${row['cost']:,.0f}  |  Financial Aid: {row['financial_aid']:.0f}%\n"
                output += f"   Admit Rate: {row['college_admit_rate']:.1%}  |  Your Chances: {row['admit_probability']:.1%} → {row['adjusted_probability']:.1%} (with ED)\n"
                output += f"   ED Fit Score: {row['ed_fit_score']:.3f}\n\n"

        self.results_text.insert(1.0, output)


if __name__ == "__main__":
    root = tk.Tk()
    app = CollegeCounselorAI(root)
    root.mainloop()