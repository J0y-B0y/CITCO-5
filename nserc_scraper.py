import pandas as pd

def exportNames(code):
    fields = ['Name-Nom', 'AreaOfApplicationCode']

    df = pd.read_csv('database.csv', encoding='latin-1', usecols=fields)

    filterBySubject = df[(df['AreaOfApplicationCode'] == code)] # 800 is the information systems code

    df_Out = filterBySubject[['Name-Nom']]

    print (df_Out) #only returns names

    df_Out.to_csv('names.csv')

print("Welcome to NSERC DB Scraper")
print("=================================")

print("By default, this program scrapes the database and return names of award winners")

print("")

code = input("Please enter the area of application code for awards you would like to retrieve. Press enter to select default code. (Information systems is 800): ")
if code == '':
    code = 800
else:
    code = int(code)

print("")

year = input("Enter the range (in fiscal years) of databases that you want to pull from: ")

print("")

print("Showing the first and last 5 entries:")

exportNames(code)

print("names.csv has been exported.")