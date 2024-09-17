import pandas as pd
import sys
def main():
    df = dataClean()
    target, features = mathRangeSetup(df)
    df_test = features.iloc[440:560]
    print("Hello, please enter a k value")
    k = int(input())
    kNearestNeighbors(features, df_test, target, df, k)

# Run KNN from 1 to k times and record the accuracy of each run
def runKNNMultipleTimes(features, df_test, target, df, k):
    accuracies = []
    for i in range(1, k + 1):
        accuracy = kNearestNeighbors(features, df_test, target, df, i)
        accuracies.append(accuracy)
    return accuracies
    
def kNearestNeighbors(features, df_test, target, df, k):
    matches = 0
    for index in range(120):
        distances = getDistances(features, df_test.iloc[index], index + 440) #defining the comparator row
        sorted_distances = sorted(distances, key=lambda x: x[1])
        neighbors = getNeighbors(sorted_distances, k)
        predicted = predictClass(neighbors, target, df)
        actual = target.iloc[440+index]
        if(predicted == actual):
            matches += 1
    accuracy = matches/120
    print("ACCURACY", accuracy)
    return accuracy
            
   
def predictClass(neighbors, classes, df): #loop through neighbors, tally classes and return most frequent class
    classesTally= {'383-400': 0, '400-420': 0, '420-440': 0, '440-460': 0,
                    '460-480': 0, '480-500': 0, '500-520': 0, '520-540': 0, '540-560': 0, '560-580': 0, '580-600': 0, '600-619': 0}
    for neighbor in neighbors: #loop through neighbors and tally classes
        rowIndex = df.index.get_loc(df[df['Total.Test-takers'] == neighbor.get('Total.Test-takers')].index[0]) #row index 
        scoreClass = classes.iloc[rowIndex]
        if(pd.isna(scoreClass)):
            continue
        classesTally[scoreClass] += 1
    max = 0
    for key, value in classesTally.items(): #find most frequent class
        if value >= max:
            max = value
            maxClass = key
    return maxClass

def dataClean():
    # Read in the data
    data_frame = pd.read_csv("school_scores.csv",
                         names=["Year","State.Code","State.Name","Total.Math",
                        "Total.Test-takers","Total.Verbal","Academic Subjects.Arts/Music.Average GPA","Academic Subjects.Arts/Music.Average Years",
                        "Academic Subjects.English.Average GPA","Academic Subjects.English.Average Years",
                        "Academic Subjects.Foreign Languages.Average GPA","Academic Subjects.Foreign Languages.Average Years","Academic Subjects.Mathematics.Average GPA",
                        "Academic Subjects.Mathematics.Average Years","Academic Subjects.Natural Sciences.Average GPA","Academic Subjects.Natural Sciences.Average Years",
                        "Academic Subjects.Social Sciences/History.Average GPA","Academic Subjects.Social Sciences/History.Average Years","Family Income.Between 20-40k.Math",
                        "Family Income.Between 20-40k.Test-takers","Family Income.Between 20-40k.Verbal","Family Income.Between 40-60k.Math",
                        "Family Income.Between 40-60k.Test-takers","Family Income.Between 40-60k.Verbal","Family Income.Between 60-80k.Math",
                        "Family Income.Between 60-80k.Test-takers","Family Income.Between 60-80k.Verbal","Family Income.Between 80-100k.Math",
                        "Family Income.Between 80-100k.Test-takers","Family Income.Between 80-100k.Verbal","Family Income.Less than 20k.Math",
                        "Family Income.Less than 20k.Test-takers","Family Income.Less than 20k.Verbal","Family Income.More than 100k.Math",
                        "Family Income.More than 100k.Test-takers","Family Income.More than 100k.Verbal","GPA.A minus.Math","GPA.A minus.Test-takers","GPA.A minus.Verbal",
                        "GPA.A plus.Math","GPA.A plus.Test-takers","GPA.A plus.Verbal","GPA.A.Math","GPA.A.Test-takers","GPA.A.Verbal","GPA.B.Math","GPA.B.Test-takers",
                        "GPA.B.Verbal","GPA.C.Math","GPA.C.Test-takers","GPA.C.Verbal","GPA.D or lower.Math","GPA.D or lower.Test-takers","GPA.D or lower.Verbal",
                        "GPA.No response.Math","GPA.No response.Test-takers","GPA.No response.Verbal","Gender.Female.Math","Gender.Female.Test-takers","Gender.Female.Verbal",
                        "Gender.Male.Math","Gender.Male.Test-takers","Gender.Male.Verbal","Score Ranges.Between 200 to 300.Math.Females","Score Ranges.Between 200 to 300.Math.Males",
                        "Score Ranges.Between 200 to 300.Math.Total","Score Ranges.Between 200 to 300.Verbal.Females","Score Ranges.Between 200 to 300.Verbal.Males",
                        "Score Ranges.Between 200 to 300.Verbal.Total","Score Ranges.Between 300 to 400.Math.Females","Score Ranges.Between 300 to 400.Math.Males",
                        "Score Ranges.Between 300 to 400.Math.Total","Score Ranges.Between 300 to 400.Verbal.Females","Score Ranges.Between 300 to 400.Verbal.Males",
                        "Score Ranges.Between 300 to 400.Verbal.Total","Score Ranges.Between 400 to 500.Math.Females","Score Ranges.Between 400 to 500.Math.Males",
                        "Score Ranges.Between 400 to 500.Math.Total","Score Ranges.Between 400 to 500.Verbal.Females","Score Ranges.Between 400 to 500.Verbal.Males",
                        "Score Ranges.Between 400 to 500.Verbal.Total","Score Ranges.Between 500 to 600.Math.Females","Score Ranges.Between 500 to 600.Math.Males",
                        "Score Ranges.Between 500 to 600.Math.Total","Score Ranges.Between 500 to 600.Verbal.Females","Score Ranges.Between 500 to 600.Verbal.Males",
                        "Score Ranges.Between 500 to 600.Verbal.Total","Score Ranges.Between 600 to 700.Math.Females","Score Ranges.Between 600 to 700.Math.Males",
                        "Score Ranges.Between 600 to 700.Math.Total","Score Ranges.Between 600 to 700.Verbal.Females","Score Ranges.Between 600 to 700.Verbal.Males",
                        "Score Ranges.Between 600 to 700.Verbal.Total","Score Ranges.Between 700 to 800.Math.Females","Score Ranges.Between 700 to 800.Math.Males",
                        "Score Ranges.Between 700 to 800.Math.Total","Score Ranges.Between 700 to 800.Verbal.Females","Score Ranges.Between 700 to 800.Verbal.Males",
                        "Score Ranges.Between 700 to 800.Verbal.Total"])
    
    # Drop the columns that are not needed
    df = data_frame.drop(["Year","State.Name", 'Total.Verbal'
                        ,"Academic Subjects.Arts/Music.Average GPA","Academic Subjects.Arts/Music.Average Years",
                        "Academic Subjects.English.Average GPA","Academic Subjects.English.Average Years",
                        "Academic Subjects.Foreign Languages.Average GPA","Academic Subjects.Foreign Languages.Average Years",
                        "Academic Subjects.Mathematics.Average Years","Academic Subjects.Natural Sciences.Average GPA","Academic Subjects.Natural Sciences.Average Years",
                        "Academic Subjects.Social Sciences/History.Average GPA","Academic Subjects.Social Sciences/History.Average Years","Family Income.Between 20-40k.Math",
                        "Family Income.Between 20-40k.Test-takers","Family Income.Between 20-40k.Verbal","Family Income.Between 40-60k.Math",
                        "Family Income.Between 40-60k.Test-takers","Family Income.Between 40-60k.Verbal","Family Income.Between 60-80k.Math",
                        "Family Income.Between 60-80k.Test-takers","Family Income.Between 60-80k.Verbal",
                        "Family Income.Between 80-100k.Test-takers","Family Income.Between 80-100k.Verbal","Family Income.Less than 20k.Math",
                        "Family Income.Less than 20k.Test-takers","Family Income.Less than 20k.Verbal","Family Income.More than 100k.Math",
                        "Family Income.More than 100k.Test-takers","Family Income.More than 100k.Verbal","GPA.A minus.Math","GPA.A minus.Test-takers","GPA.A minus.Verbal",
                        "GPA.A plus.Verbal","GPA.A.Math","GPA.A.Test-takers","GPA.A.Verbal","GPA.B.Math","GPA.B.Test-takers",
                        "GPA.B.Verbal","GPA.C.Math","GPA.C.Test-takers","GPA.C.Verbal","GPA.D or lower.Math","GPA.D or lower.Test-takers","GPA.D or lower.Verbal",
                        "GPA.No response.Math","GPA.No response.Test-takers","GPA.No response.Verbal","Gender.Female.Math","Gender.Female.Test-takers","Gender.Female.Verbal",
                        "Gender.Male.Math","Gender.Male.Test-takers","Gender.Male.Verbal","Score Ranges.Between 200 to 300.Math.Females","Score Ranges.Between 200 to 300.Math.Males",
                        "Score Ranges.Between 200 to 300.Math.Total","Score Ranges.Between 200 to 300.Verbal.Females","Score Ranges.Between 200 to 300.Verbal.Males",
                        "Score Ranges.Between 200 to 300.Verbal.Total","Score Ranges.Between 300 to 400.Math.Females","Score Ranges.Between 300 to 400.Math.Males",
                        "Score Ranges.Between 300 to 400.Math.Total","Score Ranges.Between 300 to 400.Verbal.Females","Score Ranges.Between 300 to 400.Verbal.Males",
                        "Score Ranges.Between 300 to 400.Verbal.Total","Score Ranges.Between 400 to 500.Math.Females","Score Ranges.Between 400 to 500.Math.Males",
                        "Score Ranges.Between 400 to 500.Math.Total","Score Ranges.Between 400 to 500.Verbal.Females","Score Ranges.Between 400 to 500.Verbal.Males",
                        "Score Ranges.Between 400 to 500.Verbal.Total","Score Ranges.Between 500 to 600.Math.Females","Score Ranges.Between 500 to 600.Math.Males",
                        "Score Ranges.Between 500 to 600.Math.Total","Score Ranges.Between 500 to 600.Verbal.Females","Score Ranges.Between 500 to 600.Verbal.Males",
                        "Score Ranges.Between 500 to 600.Verbal.Total","Score Ranges.Between 600 to 700.Math.Females","Score Ranges.Between 600 to 700.Math.Males",
                        "Score Ranges.Between 600 to 700.Math.Total","Score Ranges.Between 600 to 700.Verbal.Females","Score Ranges.Between 600 to 700.Verbal.Males",
                        "Score Ranges.Between 600 to 700.Verbal.Total","Score Ranges.Between 700 to 800.Math.Females","Score Ranges.Between 700 to 800.Math.Males",
                        "Score Ranges.Between 700 to 800.Math.Total","Score Ranges.Between 700 to 800.Verbal.Females","Score Ranges.Between 700 to 800.Verbal.Males",
                        "Score Ranges.Between 700 to 800.Verbal.Total"], axis=1)
    


    # Convert the State.Code to a numerical value
    df['State.Code'].replace('AL', '1', inplace=True)
    df['State.Code'].replace('AK', '2', inplace=True)
    df['State.Code'].replace('AZ', '3', inplace=True)
    df['State.Code'].replace('AR', '4', inplace=True)
    df['State.Code'].replace('CA', '5', inplace=True)
    df['State.Code'].replace('CO', '6', inplace=True)
    df['State.Code'].replace('CT', '7', inplace=True)
    df['State.Code'].replace('DE', '8', inplace=True)
    df['State.Code'].replace('FL', '9', inplace=True)
    df['State.Code'].replace('GA', '10', inplace=True)
    df['State.Code'].replace('DC', '11', inplace=True)
    df['State.Code'].replace('PR', '12', inplace=True)
    df['State.Code'].replace('HI', '13', inplace=True)
    df['State.Code'].replace('ID', '14', inplace=True)
    df['State.Code'].replace('IL', '15', inplace=True)
    df['State.Code'].replace('IN', '16', inplace=True)
    df['State.Code'].replace('IA', '17', inplace=True)
    df['State.Code'].replace('KS', '18', inplace=True)
    df['State.Code'].replace('KY', '19', inplace=True)
    df['State.Code'].replace('LA', '20', inplace=True)
    df['State.Code'].replace('ME', '21', inplace=True)
    df['State.Code'].replace('MD', '22', inplace=True)
    df['State.Code'].replace('MA', '23', inplace=True)
    df['State.Code'].replace('MI', '24', inplace=True)
    df['State.Code'].replace('MN', '25', inplace=True)
    df['State.Code'].replace('MS', '26', inplace=True)
    df['State.Code'].replace('MO', '27', inplace=True)
    df['State.Code'].replace('MT', '28', inplace=True)
    df['State.Code'].replace('NE', '29', inplace=True)
    df['State.Code'].replace('NV', '30', inplace=True)
    df['State.Code'].replace('NH', '31', inplace=True)
    df['State.Code'].replace('NJ', '32', inplace=True)
    df['State.Code'].replace('NM', '33', inplace=True)
    df['State.Code'].replace('NY', '34', inplace=True)
    df['State.Code'].replace('NC', '35', inplace=True)
    df['State.Code'].replace('ND', '36', inplace=True)
    df['State.Code'].replace('OH', '37', inplace=True)
    df['State.Code'].replace('OK', '38', inplace=True)
    df['State.Code'].replace('OR', '39', inplace=True)
    df['State.Code'].replace('PA', '40', inplace=True)
    df['State.Code'].replace('RI', '41', inplace=True)
    df['State.Code'].replace('SC', '42', inplace=True)
    df['State.Code'].replace('SD', '43', inplace=True)
    df['State.Code'].replace('TN', '44', inplace=True)
    df['State.Code'].replace('TX', '45', inplace=True)
    df['State.Code'].replace('UT', '46', inplace=True)
    df['State.Code'].replace('VT', '47', inplace=True)
    df['State.Code'].replace('VI', '48', inplace=True)
    df['State.Code'].replace('VA', '49', inplace=True)
    df['State.Code'].replace('WA', '50', inplace=True)
    df['State.Code'].replace('WV', '51', inplace=True)
    df['State.Code'].replace('WI', '52', inplace=True)
    df['State.Code'].replace('WY', '53', inplace=True)

    # Convert the columns to float from string
    df = floatConversion(df)

    # Drop the first row which has NaN values for some reason
    df = df.drop(labels=0, axis=0)

    # Drop the rows that have duplicate values for the Total.Test-takers column
    df = df.drop_duplicates(subset='Total.Test-takers', keep='first')

    # Normalize the data
    for column in df.columns: 
        df[column] = df[column]  /  df[column].abs().max() 

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def floatConversion(df):
    for columnName, series in df.items():
        df[columnName] = pd.to_numeric(df[columnName], errors='coerce')
        df[columnName] = df[columnName].astype(float)
    return df

def euclidianDistance(baseRow, iterRow, features):
    inner_value = 0
    for col in features.columns:
        inner_value += (baseRow[col] - iterRow[col]) ** 2
    return inner_value ** 0.5

def getDistances(features, curr_row, row_index):
    distances = []
    for index, row in features.iterrows():
        if index > 561:
            continue
        if index == 440:
            dist = euclidianDistance(curr_row, features.iloc[441], features)
            distances.append((row, dist))
            return distances
        if index == row_index:
            break
        dist = euclidianDistance(curr_row, features.iloc[index], features)
        distances.append((row, dist))
    return distances

def getNeighbors(sorted_distances, k):
    neighbors = []
    for x in range(k):
        neighbors.append(sorted_distances[x][0])
    return neighbors

def mathRangeSetup(df):
    df = setUpMathRanges(df)
    target = df['Total.Math']
    features = df.drop('Total.Math', axis=1)
    return target, features

def setUpMathRanges(df):
    bins = [0.6187399030694669, 0.65051158306, 0.68228326306, 0.71405494306, 0.74582662306, 0.77759830306, 0.80936998306, 0.84114166306, 0.87291334306, 0.90468502306, 0.93645670306
            , 0.96822838306, 1.0]
    names = ['383-400', '400-420', '420-440', '440-460', '460-480', '480-500', '500-520', '520-540', '540-560', '560-580', '580-600', '600-619']
    df['Total.Math'] = pd.cut(df['Total.Math'], bins, labels=names)

    return df

def testEuclidianOne(df):
    row1 = df.iloc[1]
    row2 = df.iloc[20]
    distance = euclidianDistance(row1, row2, df)
    print(distance)

def testGetNeighborsOfRowOne(features):
    row1 = features.iloc[1]
    distances = getDistances(features, row1, 1)
    sorted_distances = sorted(distances, key=lambda x: x[1])
    for dist in sorted_distances:
        print(dist[1])

def testGetNeighborsOfRow571(features): #found bug here with Virgin Islands
    row1 = features.iloc[1]
    distance = euclidianDistance(row1, features.iloc[571], features)
    print(distance)

if __name__ == '__main__':
    main()