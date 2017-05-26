###############################################
#     Shuo Yuan (Steve) Yang, Joseph Lee      #
###############################################

import sys
from random import randint
import copy
import csv
import math


##################################################################
#   takes in the data set                                         #
#   determine if the input data is csv or data, as there          #
#   are different ways to take them as input                      #
#                                                                 #
################################################################## 



userInputData = sys.argv[1]
userInputK = int(sys.argv[2])
outputName = sys.argv[3]


linesForData = []



if '.csv' in userInputData :
	with open(userInputData, 'rb') as f:
		reader = csv.reader(f)

		for elem in reader:
			line = list(map(float, elem[2:]))
			linesForData.append(line)

	data = linesForData
	
	
else:
	with open(userInputData) as file:
		for line in file.readlines():
			line = [int(each) for each in line.split(',')[2:]]
			linesForData.append(line)

	data = linesForData


##################################################################
#   Take in two points in the argument                            #
#   return the distance between 2 points                          #
#                                                                 #
################################################################## 

def distance (p1, p2):
	square = 0
	for i, val in enumerate(p1):
		square += (p1[i]-p2[i])**2
	
	return math.sqrt(square)

##################################################################
#                                                                 #
#   This method is used in the very beginning of the algorithm    #
#   Take in the Data and the k, speified by the user              #
#   generate and return a list of k number of ranomly             #
#   chosen centroids                                              #
#                                                                 #
################################################################## 


def randomCentroid(data, k):     ##initial random centroids

	times = 0

	listOfrandomCentroid = []


	while (times < k):

		random = randint(0,len(data)-1)
		listOfrandomCentroid.append(data[random])
		times+=1


	return listOfrandomCentroid

##################################################################
#                                                                 #
#   This method is used in the very beginning of the algorithm    #
#   It is used for minimizing the given number                    #
#   by taking in the min max value                                #
#                                                                 #
################################################################## 


def calculate(toBeNormalized, mina, maxa, new_mina, new_maxa):
	upper = float(toBeNormalized - mina)
	
	under = float(maxa - mina)
	
	minus = float(new_maxa- new_mina)
	
	first = float((upper/under)*minus)
	
	answer = float(first + new_mina)
	return answer

##################################################################
#                                                                 #
#   This method is used in when constructing the MinMax list.     #
#   It takes in the dataset and the column number,                #
#   and return the minium value of that specified column          #
#                                                                 #
################################################################## 

def findMin(dataset, column):
	minimum = 9999999
	for row in dataset:
		if row[column] < minimum:
			minimum = row[column]
	return minimum

##################################################################
#                                                                 #
#   This method is used in when constructing the MinMax list.     #
#   It takes in the dataset and the column number,                #
#   and return the maximum value of that specified column         #
#                                                                 #
################################################################## 

def findMax(dataset, column):
	maximum = 0
	for row in dataset:
		if row[column] > maximum:
			maximum = row[column]
	return maximum

##################################################################
#                                                                 #
#   This method is used when normalizing the data.                #
#   It takes in the dataset,                                      #
#   and return a list that has the min and max value of each      #
#   attribute.                                                    #
#                                                                 #
################################################################## 

def constructMinMaxList(dataset):
	
	minMaxList = []

	for index, val in enumerate(dataset[0]):
		temp = []
		
		temp.append(findMin(dataset,index))
		
		temp.append(findMax(dataset,index))
		minMaxList.append(temp)

	return minMaxList


##################################################################
#                                                                 #
#   This method is used when normalizing the data.                #
#   It takes in the dataset,                                      #
#   and return a normalized dataset for clustering                #
#                                                                 #
################################################################## 


def normalize(dataset):
	newDataset = []
	minMaxList = constructMinMaxList(dataset)
	for row in dataset:
		temp = []
		for index,elem in enumerate(row):
			
			temp.append(calculate(elem, minMaxList[index][0], minMaxList[index][1],0.0,1.0))
		newDataset.append(temp)

	return newDataset

##################################################################
#                                                                 #
#   This method is used for calculating the average of the        #
#   cluster, which will be used as the centroid for the next      #
#   iteration.                                                    #
#   It takes in the dataset, and the cluster, it will first check #
#   how many points in the cluster, excluding the centroid of     #
#   that cluster.                                                 #
#   If it does not have any point, it will randomly               #
#   generate a centroid by randomly choosing a point in the data  #
#                                                                 #
#   If it only has 1 point for the cluster, it will just return   #
#   that one point as the next centroid                           #
#                                                                 #
#   Else, calculatethe average mean of the whole cluster and      #
#   return that as the centroid                                   #
#                                                                 #
################################################################## 


def clusterMean (data, cluster):
	numberOfPoints = len(cluster)-1
	
	if numberOfPoints == 0:
		
		randomNumber = randint(0, len(data)-1)
		a = data[randomNumber]
		toBeReturned = a
		return toBeReturned

	if numberOfPoints == 1:
		
		toBeReturned = cluster[1]
		return toBeReturned

	temp = copy.deepcopy(cluster[1])

	toBeReturned = []
	for point in cluster[2:]:
		for index, num in enumerate(point):
			temp[index]+= num

	for num in temp:
		avg = round(num,5)/numberOfPoints
		toBeReturned.append(avg)

	return toBeReturned


##################################################################
#                                                                 #
#   The 'main' method of the kmeans algorithm.                    #
#                                                                 #
#                                                                 #
#   It takes in the dataset, and the k input by the user.         #
#   It first normalizes the dataset to be based on 0.0 to 1.0     #
#                                                                 #
#   It will return a list of k clusters                           #
#                                                                 #
################################################################## 



def kMeans(dataset, k):    ##changed
	normalizedData = normalize(dataset)
	
	centroids = randomCentroid(normalizedData, k) ##a list of randomly chosen centroid points

	
	hasChange = True #stopping condition for the while loop

	normalizedCopy = copy.deepcopy(normalizedData)

	
	itera = 0

	oldClusters = [] #this is for storing the cluster for comparing between the new and old clusters
					 #it will also be returned as the clustering
	

	while hasChange == True:
		

		clusters = []    #a list of current cluster

		for each in centroids:   #this is to assign each centroid as the head of each cluster
			temp = []
			temp.append(copy.deepcopy(each))
			clusters.append(copy.deepcopy(temp))




		for point in normalizedCopy:  # Each each point from the dataset will be taken out

			
			clusterNum = 0
			min_distance = 9999

			## that point taken out will be compared to centroid of each cluster
			## and be assigned to the cluster that is the closest to the centroid.
			for index, c in enumerate(centroids):  

				

				cDistance = distance(point, c)
			
				
				if cDistance < min_distance:
				
					min_distance = cDistance
					clusterNum = index
			
			
			clusters[clusterNum].append(copy.deepcopy(point))



		



		
		
	    #generating a new centroid set for the next iteraton
	    #by calculating the means of each cluster, and that mean
	    #would be used for the new centroid of that cluster
		newCentroidSet = []

		for cluster in clusters:
			
			newCentroid = clusterMean(normalizedCopy, cluster)
			newCentroidSet.append(copy.deepcopy(newCentroid))
			
	
		#stopping condition is true, assign that cluster as oldCluster and then break out of the loop
		if stopCondition(oldClusters, clusters, k) == True:
			oldClusters = clusters
			print itera
			break;


		oldClusters = clusters
		centroids = newCentroidSet
		itera +=1 

	return oldClusters


##################################################################
#                                                                 #
#   This method is used in when calculating the SSE of each       #
#   cluster.                                                      #
#   It takes in the dataset and the mean of that centroid,        #
#   and return the SSE of that cluster.                           #
#                                                                 #
################################################################## 


def clusterSSE(cluster, cMean):
	SSE = 0
	for point in cluster:
		SSE += distance(point, cMean)**2

	return SSE




##################################################################
#                                                                 #
#   This method is used in determining if the stopping            #
#   condition is met.                                             #
#   It takes in the current newly generated cluster, previously   #
#   generated cluster, and k                                      #
#   and it returns whether or not the clusters are the same       #
#                                                                 #
################################################################## 


def stopCondition(clusters, newClusters, k):
	sameCluster = False
	count = 0
	for index, val in enumerate(clusters):
		if clusters[index] == newClusters[index]:
			count += 1
	
	if count == k:
		sameCluster = True

	return sameCluster




		

clusterFinal = kMeans(data,userInputK)





fw = open(outputName,"w")
totalSSE = 0
for index, cluster in enumerate(clusterFinal):
	SSE = clusterSSE(cluster[1:], cluster[0])
	totalSSE += SSE
	fw.write("Cluster  " +str(index)+" :  "+ str(cluster) + '\n'+'\n'+"SSE of cluster "+str(index)+" Computed:  " + str(SSE) +'\n' + 'number of points in cluster: '+str(len(cluster)-1)+'\n')
	fw.write('\n')
	fw.write('\n')
	fw.write('\n')
fw.write('Total SSE is: ' + str(totalSSE))

fw.close()





