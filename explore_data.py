import pandas as pd

# Load the college data
colleges_df = pd.read_csv('college_admissions.csv')

print("\n" + "="*80)
print("AVAILABLE COLUMNS IN YOUR DATASET")
print("="*80 + "\n")

# Group columns by category
columns = colleges_df.columns.tolist()

print("ACADEMIC METRICS:")
academic = [col for col in columns if any(x in col.lower() for x in ['sat', 'act', 'gpa', 'score', 'percentile'])]
for col in academic:
    print(f"  - {col}")

print("\nENROLLMENT & SIZE:")
enrollment = [col for col in columns if any(x in col.lower() for x in ['enrollment', 'students', 'total'])]
for col in enrollment[:10]:  # First 10
    print(f"  - {col}")

print("\nFINANCIAL:")
financial = [col for col in columns if any(x in col.lower() for x in ['tuition', 'price', 'aid', 'grant', 'loan', 'endowment'])]
for col in financial:
    print(f"  - {col}")

print("\nDEMOGRAPHICS:")
demographics = [col for col in columns if any(x in col.lower() for x in ['percent', 'enrollment that', 'american', 'asian', 'black', 'hispanic', 'white', 'women', 'race'])]
for col in demographics[:15]:  # First 15
    print(f"  - {col}")

print("\nINSTITUTION TYPE:")
institution = [col for col in columns if any(x in col.lower() for x in ['sector', 'control', 'level', 'carnegie', 'tribal', 'hbcu', 'urbanization'])]
for col in institution:
    print(f"  - {col}")

print("\nOTHER:")
other = [col for col in columns if col not in academic + enrollment + financial + demographics + institution]
for col in other:
    print(f"  - {col}")

print("\n" + "="*80)
print("SAMPLE VALUES FOR CATEGORICAL COLUMNS")
print("="*80 + "\n")

# Show unique values for categorical columns
if 'Sector of institution' in colleges_df.columns:
    print("Sector of institution:")
    print(colleges_df['Sector of institution'].unique())
    print()

if 'Control of institution' in colleges_df.columns:
    print("Control of institution:")
    print(colleges_df['Control of institution'].unique())
    print()

if 'Level of institution' in colleges_df.columns:
    print("Level of institution:")
    print(colleges_df['Level of institution'].unique())
    print()

if 'Historically Black College or University' in colleges_df.columns:
    print("Historically Black College or University:")
    print(colleges_df['Historically Black College or University'].unique())
    print()

if 'Degree of urbanization (Urban-centric locale)' in colleges_df.columns:
    print("Degree of urbanization (Urban-centric locale):")
    print(colleges_df['Degree of urbanization (Urban-centric locale)'].unique())
    print()

print("\n" + "="*80)
print("SAMPLE COLLEGE DATA")
print("="*80 + "\n")
print(colleges_df[['Name', 'Sector of institution', 'Control of institution', 'Level of institution', 'Percent of total enrollment that are women']].head(10))