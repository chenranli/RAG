
# Read numbers from file and calculate average
total = 0
count = 0

file_path = 'scores.txt'
# Write to file

with open(file_path, 'r') as file:
    for line in file:
        number = float(line.strip())  # Convert string to integer
        total += number
        count += 1

# Calculate average
if count > 0:
    average = total / count
    print(f"The average is: {average}")
else:
    print("No numbers found in the file.")


