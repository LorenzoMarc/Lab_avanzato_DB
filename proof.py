import re
import pandas as pd

# Define the text from which you want to extract the data
text = '''
[AtlMi:] [VP234: VO45: VP4B56] [APPS:]
IF 
TBM receives a valid Remote Unlock request command from the telematic client
AND 
TBM has received U_CONN_SEED message with Seed LIDs ($Uconseed$, $uconseed2$)
AND 
$vehicleSpeed$ >=[3 km/h]
AND 
TBM shall check for the status of below signals and if one of them is not equal to [Closed]
-$DriverDoorSts$
-$LHRDoorSts$
THEN
TBM shall:
- set $UCOnn$ = [NO_BASIC_REQUEST]'
'''

# Regular expressions to extract data
architecture_regex = r"\[(AtlMi|AtlHi|APPS)\]"
message_regex = r"TBM (?:receives|has received) ([\w\s]+) message"
lid_regex = r"\$([\w\s]+)\$"
value_regex = r"\[([\w\s]+)\]"

# Extract architecture
architecture_match = re.search(architecture_regex, text)
architecture = architecture_match.group(1) if architecture_match else ""

# Extract messages, LIDs, and values
messages = re.findall(message_regex, text)
lids = re.findall(lid_regex, text)
values = re.findall(value_regex, text)

# Prepare the data for the Excel file
data = {
    "Architecture": [architecture] * len(messages),
    "Message": messages,
    "LID": lids,
    "Value": values
}
print(data)
exit()
# Create a DataFrame to hold the extracted data
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
filename = "extracted_data.xlsx"
df.to_excel(filename, index=False, engine="openpyxl")
print("Data extracted and saved to", filename)
