'''
	Sam Agriculture and Artificial Intelligence System

		This automated agriculture system can utilize known data to 
	create educated options to chose from: These options are selected
	utilizing a neural network to estimate plant growth height. The
	neural network is trained and tested to be able to simulate plant
	growth rates based on a variety of inputs. It will utilize real
	plant data to generate a complex arbitrary math function to
	calculate an estimate plant height.
	
		Along with using an Artificial Intelligence to allow the 
	computer to understand the affect of each input on the plant,
	this system also employs a Darwin-esk survival of the fittest.
	To prevent missing potential benefitial outlier regions and to
	encourage testing all potential input options there is a gradient
	of selection. This gradient prefers input values that worked well
	in growth but still allow room for testing more varied options.
		The artificial intelligence and survival of the fittest models
	are additive to a selective breeding process. Prior to user input,
	the Artifical Intelligence is called upon again to simulate the
	outcome of all possible input combinations (*all numerics can be
	adjusted by the user at any time or not at all*):
	
		For 9* inputs given 5* potential options selected from the
	survival of the fittest/luckiest function there are 1,953,125
	(5^9) possible options to chose from. This obviously is way too
	many to chose from as the user. The Artifical Intelligence comes
	back into play, simulating the potential plant growth height given
	each set of inputs for all 1,953,125. After quick sorting the best
	20* options are shown to the user. These options can have a common
	constant input (very biased/productive input or accurate estimated
	input). The user can then chose which of the best possible choices
	to chose from. This method allows specific testing of fewer 
	variables at a time allowing more accurate data for improving the
	Artificial intelligence. It still is robust enough to test varied
	options yet accurate enough to retain important input values.

		Given plant input selections, the user can opt to grow the
	plant choices and record the data using an automated growth system.
	The growth system varies the environment to follow each chosen
	input level and records plant growth over time. This allows the
	system to be almost fully automated from start to finish.

		Using the real plant data recieved from the grown plants, the
	Artifical Intelligence uses the new data along side all previous
	data to improve it's ability to estimate plant growth. With the
	AI improving, it will continue to make more educated input choices
	(short/long term), better survival of the fittest (long term), and
	better selections for the user to chose from. Over time the AI will
	become refined in a plant's attributes and how the inputs affect
	growth. This will allow accurate judgement of a plant's ideal
	environment and provide much better plants!


	Table of Contents (Updated 7/25/17)
	'$' = Completed		'>' = Future Changes

		- 1 - Data Storage ~ Line 160
				$ Stores data to writable CSVs
				> Stores long-term data to local SQL server

		- 2 - Collect Data From Sensors ~ Line 263
				$ Can collect various input data
				> Physical system to analyze multiple plants
				> Inputs can be dynamic (physical system)

		- 3 - Suggesting New Inputs ~ Line 435
				$ Uses darwin-esk survival of the fittest
				$ Allows for selective breeding by user
				$ Allows basic computer choice by best
				> Allow the computer to make decisions based off unique/productive data

		- 4 - Artificial Intelligence for Analysis of Plant Growth ~ Line 649
				$ Can use simulated or real plant growth data
				$ Can create predictions of values soley based off inputs
				$ Can get value to different suggested input choices (from section 3)
				$ Can train for a user set number of cycles and from a user set quantity of data
				$ Can be expanded to use more or less neural layers and neurons in each layer
				$ Automatically adjusts internal neural math network to fit a plant growth probability
				$ Data Visualization through Tensorflow (tensorboard)
				> Improve accuracy with less data (smarter analysis)

		- 5 - Simulated plant growth ~ Line 914
				$ Basic math system to create 'real' simulated data to check AI's ability
				$ Create full cycle of ' Choices -> Planting -> Data Collection -> AI -> Choices '
				$ AI won't have access to the 'real' math function but will have to mimic it's ability
				> Better UI

		- 0 - Main / Menu
				$ Simple system to test all previous sections
				$ Create a better UI system in console
				 >Get physical LCD menu operating to run apart from a computer console
				> Send data/recieve from long-term data sources (SQL)
				> Create webpage
'''

import csv, sys, time, os, math, glob

import argparse

import pandas as pd
import tensorflow as tf
import numpy as np

from os import listdir

'''
import pyupm_grove as grove
import pyupm_i2clcd as lcdi2c		#LCD screen
import pyupm_htu21d as hti2c		#Temp & Humidity
import pyupm_biss0001 as motion 	#Motion
import pyupm_buzzer as buzzer 		#Sound output
import pyupm_mic as mic 			#Sound sensor
import pyupm_servo as servo 		#Servo motor
import pyupm_grovemoisture as upmMoisture
import pyupm_guvas12d as upmUV
'''

tf.logging.set_verbosity(tf.logging.INFO)

global standard
global robust

# Section 1: Write to CSV ----------------------------------------------------

''' #Real Data:
COLUMNS = ["Date", "Time", "TestNumber", "AirTemp", "LightLevel",
            "Humidity", "Motion", "PlantGrowth"]
FEATURES = ["Date", "Time", "TestNumber", "AirTemp", "LightLevel",
            "Humidity", "Motion"]
LABEL = ["PlantGrowth"]
'''

#Simulated Data:
COLUMNS = ["PlantingDepth", "SoilAcidity", "WaterVapor", "LiquidWater", "CO2",
			"HeatPad", "RedLight", "BlueLight", "UVLight", "PlantGrowth"]
FEATURES = ["PlantingDepth", "SoilAcidity", "WaterVapor", "LiquidWater", "CO2",
			"HeatPad", "RedLight", "BlueLight", "UVLight"]
LABEL = ["PlantGrowth"]

directory = 'C:/Users/shanson/Desktop/Python/AgricultureAIAutomation/AI_Data/PlantGrowth/'
originalFile = 'InitialData.csv'			#Starting data to work with
recordFile = 'PlantData.csv'			#Input sensor Data
potentialFile = 'PotentialChoices.csv'	#Predicted input choices
simFile = 'SimulatedData.csv'
inputSetsFile = 'InputSets.csv'
compareFile = 'ComparePredictToReal.csv'


#Remove the file to create a new blank csv
def refreshFile(direct, file):
	if os.path.exists(direct + file):
		print('Deleting file: ' + str(direct) + str(file))
		os.remove(direct + file)
	else:
		print('File not found')
	#If no file, create it and set category header
	if not os.path.exists(direct + file):
		print('Creating file: ' + str(direct) + str(file))
		with open(direct + file, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			#writer.writerow(COLUMNS)
		csvfile.close()
	else:
		print('File already exists: '+ str(direct) + str(file))


# Section 1: Data Storage ---------------------------------------------------------------------
#Pretty much substring creator
def replacer(oldstring, add, index, nofail=False):
	if index < 0:
		return add + oldstring
	if index > len(oldstring):
		return oldstring + add
	newstring = oldstring[:index] + [str(add)] + oldstring[index + 1:]
	return newstring

#Add data to the earliest open slot
def addDataInLine(category, value, direct, file):
	print('Add data: ' + category)
	index = 0
	for x in COLUMNS:
		if category == x:
			break
		else:
			index = index + 1

	with open(direct + file) as csvfile:		#newline=''
		reader = csv.reader(csvfile)
		rowlist = []
		try:

			added = False
			numTest = 0
			for row in reader:

				if numTest > 0:
					row[2] = str(numTest)
				numTest = numTest + 1

				if row[index] == '-' and not added:

					newrow = replacer(row, value, index)
					testrow = ','.join(newrow).replace('|', '')
					added = True
					rowlist.append(testrow)

				else:

					rowlist.append(','.join(row).replace('|', ''))

			if not added:

				curdate = str(time.strftime('%d-%m-%Y'))
				curtime = str(time.strftime('%H:%M'))

				blankrow = '-,-,-,-,-,-'
				if index == 0:
					blankrow = str(value)+blankrow[1:]
				else:
					blankrow = blankrow[0:index*2-2] + str(value) + blankrow[index*2-3:]
				blankrow = curdate + ',' + curtime +  blankrow

				rowlist.append(blankrow)

		except csv.Error as e:
			sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))

	with open(direct + file, 'w') as csvfile:	#ewline=''
		writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for numb in rowlist:
			writer.writerow([numb])
	csvfile.close()

#Lock all tests, so that new tests can be added
def lockPastTests(direct, file):
	if robust:
		print('Locking past tests')
	rowlist = []
	with open(direct + file) as csvfile:  	#newline=''
		reader = csv.reader(csvfile)
		try:
			for row in reader:  

				x = 0
				while x < len(row):
					if row[x] == '-':
						row[x] = '~'
					x = x + 1
				if len(row) > 0:
					rowlist.append(','.join(row).replace('|', ''))
					#print(rowlist)

		except csv.Error as e:
			sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
	csvfile.close()

	with open(direct + file, 'w') as csvfile:	 #newline=''
		writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for numb in rowlist:
			writer.writerow([numb])
	csvfile.close()

#Create a new test level and add some data
def addNewData(category, value, direct, file):
	print('Adding new data')
	lockPastTests(direct, file)
	addDataInLine(category, value, direct, file)



# Section 2: Collect Data From Sensors --------------------------------------------------------

#I2C Sensors/Displays
'''
lcd = lcdi2c.Jhd1313m1(0, 0x3E, 0x62)		#LCD display on I2C
humidtemp = hti2c.HTU21D(6, 0x40) 			#Temp & Humidity sensor on I2C

#Digital Sensors/Inputs/Outputs
buzzer = buzzer.Buzzer(2)					#Sound output on D2
button = grove.GroveButton(3)				#Button input on D3
motion = motion.BISS0001(7)					#Motion sensor on D7

#Analog Sensors/Inputs
temp = grove.GroveTemp(0) 					#Grove temperature on A0
rotary = grove.GroveRotary(1)				#Grove angle sensor on A1
moisture = upmMoisture.GroveMoisture(1) 	#Moisture sensor on A1
light = grove.GroveLight(2) 				#Grove light on A2
servo = servo.ES08A(3)						#Servo motor on A3
UV = upmUV.GUVAS12D(3) 						#UV sensor on A3
mic = mic.Microphone(3)						#Sound sensor on A3

humidtemp.resetSensor()

#Outputs and Display --------------
#Display on LCD Screen
def displayLCD(line1, line2):
	lcd.clear()
	lcd.setColor(255,255,0)

	lcd.setCursor(0,0)
	lcd.write(line1)
	lcd.setCursor(1,0)
	lcd.write(line2)

#Buzzer Sound Creation
def playSound(duration = 1, volume = 0.25, tone = DO):
	buzzer.setVolume(volume)
	buzzer.playSound(tone, 1000000*duration)

def getVolume():
	return buzzer.getVolume()

def playConstantSound(volume = 0.25, tone = DO):
	buzzer.setVolume(volume)
	buzzer.playSound(tone)

def stopSound():
	buzzer.stopSound()

#Servo Movement
def moveServo(angle):							#Value of 0 to 180 in Degrees
	servo.setAngle(angle)

#Microphone Collection
def getSounds(numSamples = 1, offPeriod = 1, buffer = 1):
	return mic.getSampledWindow(offPeriod, numSamples, buffer)

#Data Collection --------------
#Temperature Collection
def docTemp():
	tempV = temp.value()
	addDataInLine('AirTemp', tempV)
	return tempV

def getTemps(quantity, period):
	while quantity > 0:
		docTemp()								#In Celsius
		quantity = quantity - 1
		time.sleep(period)

def getTemps(quantity, period):
	while quantity > 0:
		docTemp()
		quantity = quantity - 1
		time.sleep(period)
#Light Collection
def docLight():
	lightV = light.value()
	addDataInLine('LightLevel', lightV)
	return lightV

def getLights(quantity, period):
	while quantity > 0:
		docLight()
		quantity = quantity - 1
		time.sleep(period)

#Humidity Collection
def docHumidity():
	humidtemp.resetSensor()
	humidity = humidtemp.getHumidity()			#Measured humidity (RH)
	#print(humidity)
	addDataInLine('Humidity', humidity)
	return humidity

def getHumidities(quantity, period):
	while quantity > 0:
		docHumidity()
		quantity = quantity - 1
		time.sleep(period)

#Extra Humidity Functions
def docHumTemp():
	humidtemp.resetSensor()
	humTemp = humidtemp.getTemperature()		#Humidity cell temperature (Celcius)
	#print(humTemp)
	return humTemp

def compHumidity():
	humidtemp.resetSensor()
	compRH = humidtemp.getCompRH()				#Calculated compensated RH
	#print(compRH)
	return compRH
	
def dewPoint():
	humidtemp.resetSensor()
	dewPoint = humidtemp.getDewPoint()			#Dew point (Celcius)
	#print(dewPoint)
	return dewPoint

#Motion Collection
def docMotion():
	movement = motion.value()
	addDataInLine('Motion', motion)
	return movement

def getMotions(quantity, period):
	while quantity > 0:
		docMotion()
		time.sleep(period)
		quantity = quantity - 1

#CollectAllData --------------
def collectAllData():
	print('Collecting all data')
	lockPastTests()
	lightV = docLight()
	tempV = docTemp()
	displayLCD('Temp: ' + str(tempV), 'Light: ' + str(lightV))
	humidity = docHumidity()
	motion = docMotion()

def collectMultipleSets(quantity, period):
	while quantity > 0:
		collectAllData()
		time.sleep(period)
		quantity = quantity - 1

#Rotary Angle input
def rotaryAngle(mtype = 'degs'):					#accepts rads & degs
	if mtype == 'degs':
		angle = rotary.abs_deg()
	elif mtype == 'rads':
		angle = rotary.abs_deg()
	else:
		angle = -1
		print('Error: Please use rads or degs to return rotary angle')
	return angle

#Soil moisture
def getSoilMoisture():
	return moisture.value()

#Ultra Violet
def getUV():
	#voltage = UV.volts()
	UV = UV.intensity()
	return UV
'''



# Section 3: Suggesting New Inputs ------------------------------------------------------------

def createList(direct, file):
	#print('Creating list from ' + str(direct) + str(file))
	listed = pd.read_csv(direct+file, skipinitialspace=True, skiprows=0, names=COLUMNS)			#used to develop predictions
	#print(listed)
	return listed 

def formMatrix(pandaList):
	if robust:
		print("Developing panda matrix")
	with tf.name_scope('suggestInputs'):
		#Training Data
		input_matrix = pandaList.as_matrix(columns=FEATURES)
		output_matrix = pandaList.as_matrix(columns=LABEL)
	return input_matrix, output_matrix

def inVSout(inputMatrix, outputMatrix, inputTitles, category = 'AirTemp', whichSort = 'out'):		#whichSort = 'in' OR 'out'
	#print('Input Matrix:\n' + str(inputMatrix))
	index = 0
	found = False
	while index < len(inputTitles):
		if inputTitles[index] == category:
			found = True
			break
		index = index + 1

	if found:		
		if robust:
			print('Category Input Index: ' + str(index))
		soloInput = np.reshape(inputMatrix[:,index], (-1,1))
		#print('Solo Input Length:\n' + len(soloInput))		#Should be 500 for each line of the inputs
		#print('Solo Input Array:\n'+ str(soloInput))
		#print('Solo Output Array:\n'+ str(outputMatrix))
		inAndout = np.append(soloInput, outputMatrix,axis=1)
		#inAndout = np.concatenate((soloInput, outputMatrix), axis=1)
		#print('Concatenated In and Out:\n' + str(inAndout))]
		inAndout = np.around(inAndout.astype(float), 4)
		#print(inAndout)

		if whichSort == 'out':
			sortedInput = inAndout[inAndout[:,1].argsort(),:]
		else:
			sortedInput = inAndout[inAndout[:,0].argsort(),:]
		#print('Sorted by Output:\n' + str(sortedInput))
	else:
		print(str(category) + ' not found!')
		sortedInput = False
	return sortedInput

def getBest(sortedInOut, percentage):		#percentage: 0 to 100 (not decimal)
	quantity = len(sortedInOut[:,0])
	indexStart = int(quantity - quantity*percentage/100)
	topPercent = sortedInOut[indexStart:, :]
	#print(topPercent)
	#print(len(topPercent))
	return topPercent

def suggest(sortedInTop, numGroups):
	totalQuantity = len(sortedInTop)
	numPerGroup = int(totalQuantity/numGroups)
	if robust:
		cin = input('Press enter to proceed')
	os.system('cls')
	if robust:
		print('Top ' + str(totalQuantity) + ' (sorted by input): ')
		print(sortedInTop)

	#print('Quantity/Group: ' + str(numPerGroup))
	leftOvers = totalQuantity - numPerGroup*numGroups
	#print('Quantity Leftover: ' + str(leftOvers))
	
	#Initalize groups:
	groups = []
	chosenGuesses = []
	#print(groups)
	x = 0
	while x < numGroups:
		groups.append(0)
		x = x + 1
	#print(groups)
	#Count total number:
	while leftOvers > 0:
		rand = int(np.random.rand()*numGroups)
		#print(rand)
		if len(groups)-1 < rand:
			groups[rand] = 1
		if groups[rand] == 1:
			print('')
			#print('Num repeated')
		else:
			groups[rand] = 1
			leftOvers = leftOvers - 1
	x = 0
	startLevel = 0
	while x < numGroups:
		#print(groups[x])
		quantityInGroup = groups[x] + numPerGroup
		groups[x] = sortedInTop[startLevel:quantityInGroup+startLevel][:]
		#print(groups[x])
		#print('Start Level: '+str(startLevel)+'|QuantityInGroup: '+str(startLevel+quantityInGroup))
		startLevel = startLevel + quantityInGroup
		oneGroup = groups[x][:,0]
		maxGroup = np.max(oneGroup)
		minGroup = np.min(oneGroup)
		ranValInRange = minGroup + np.random.rand()*(maxGroup-minGroup)
		groups[x] = ranValInRange
		x = x + 1
	#print(groups)
	return groups

def suggestInputs(category, numChoices, percentageBest, direct, file):		#percentageBest = 0 to 100
	listy = createList(direct, file)
	matrixInput, matrixOutput = formMatrix(listy)
	titleInMatrix = matrixInput[0]
	matrixInput, matrixOutput = matrixInput[1:], matrixOutput[1:]
	#print('matrix[0]: ' + str(matrixInput[0]) + str(matrixOutput[0]))	#Doesn't include titles

	sortedVapor = inVSout(matrixInput, matrixOutput, titleInMatrix, category)
	#print(len(sortedVapor))
	if robust:
		print('Sorted By Output List:\n' + str(sortedVapor))
	if type(sortedVapor) is bool:
		goMo = False
	else: 
		goMo = True

	if goMo:	
		topVapor = getBest(sortedVapor, percentageBest)
		sortInTopVapor = topVapor[topVapor[:,0].argsort(),:]
		#print(sortInTopVapor)
		#print(np.mean(sortInTopVapor))
		if robust:
			print('Above is ' + str(category) + ' list sorted by output for input selection.')
		choices = suggest(sortInTopVapor, numChoices)
		if  robust:
			print('Suggested ' + str(category) + ' Choices: \n' + str(choices))
			wait = input('Press enter to proceed.')
			os.system('cls')

	else:
		print('Cant suggest choices due to no data to chose from.')
		choices = False
	return choices

def getAllSuggestions(numEach, helpProb, direct, file):
	suggested = []
	nonDataCols = 0
	intCat = 0
	for x in FEATURES:
		suggest = suggestInputs(x, numEach, helpProb, direct, file)
		if suggest != False:
			suggested.append(0)
			suggest.insert(0,x)
			suggested[intCat-nonDataCols] = suggest
			intCat = intCat + 1
	if standard:
		print('Full suggested list:')
		for each in suggested:
			print(each)
		wait = input('Press enter to proceed')
		os.system('cls')

	if suggested == []:
		suggested = False
		print('NO DATA FOUND, Cannot execute creating choices.')
		wait = ('Press enter to proceed.')
		os.system('cls')
	return suggested

def createChoices(listedData, direct, file):
	if standard:
		print('Adding all data to ' + str(direct) + str(file))
	lockPastTests(direct, file)
	#print(listedData)
	numGroups = len(listedData)
	eachType = listedData[0]
	numData = len(eachType) - 1
	totalChoices = np.power(numData, numGroups)
	if standard:
		print('\n# of Groups: ' + str(numGroups))
		print('# of Data: ' + str(numData))
		print('Total # of Choices: ' + str(totalChoices))
		wait = input('Press enter to proceed.')
		os.system('cls')
	x = 0

	#Assess "binary" multiplier to create even transition from 0,0,0,0 to 0,0,0,1 ... to 0,0,1,0 etc.
	numericLine = []
	while x < numGroups:
		numericLine.append(0)
		x = x + 1

	x = numGroups - 1
	y = 1
	while 0 <= x:
		numericLine[x] = y
		x = x - 1
		y = y * numData

	#Fill the options based off of the binary output, resulting in every possible option
	x = 0
	options = []
	titles = []
	while x < numGroups:
		oneGroup = listedData[x]
		titles.append(oneGroup[0])
		x = x + 1

	options.append(titles)

	listedData[0]
	while x < totalChoices:
		newList = []
		val = x
		for y in numericLine:
			newList.append(int(val/y))	
			val = val - int(val/y)*y
		#print(newList)
		index = 0
		for z in newList:
			group = listedData[index]
			#print(group)
			newList[index] = group[z+1]
			index = index + 1
		options.append(newList)
		#print(newList)
		x = x + 1
	#print(options)
	#print(str(len(options))+'/'+str(totalChoices))

	with open(direct + file, 'w') as csvfile:		#newline=''
		writer = csv.writer(csvfile, delimiter=',', quotechar='|', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
		writer.writerows(options)
	csvfile.close()
	return totalChoices, options



# Section 4: AI to Predict Plant Height -------------------------------------------------------

def train(direct, file, numCycles, batchSize, numChoices = 1, directChoices = False, fileChoices = False):
	if standard:
		print('\nStarting Training Session')

	# Load datasets
	training_set = pd.read_csv(direct + file, skipinitialspace=True, skiprows=1, names=COLUMNS)
	
	predictable = False
	if type(directChoices) is not bool and type(fileChoices) is not bool:
		print('Prediction set ready.\n')
		prediction_set = pd.read_csv(directChoices + fileChoices, skipinitialspace=True, skiprows=1, names=COLUMNS)
		predictable = True
	else:
		print('No predictions available, just training.\n')

	sess = tf.Session()

	if standard:
		print('\nTensorflow is ready to learn.')
		wait = input('Press enter to begin training.')

	os.system('cls')


	#Train from a smaller batch of all data:
	def next_batch(num, data, labels):
		np.set_printoptions(suppress=True)
		idx = np.arange(0, len(data))
		np.random.shuffle(idx)
		idx = idx[:num]
		data_shuffle = [data[ i] for i in idx]
		labels_shuffle = [labels[ i] for i in idx]
		return np.asarray(data_shuffle), np.asarray(labels_shuffle)

	def org_batch(data, labels):
		np.set_printoptions(suppress=True)
		return np.asarray(data), np.asarray(labels)


	#Organize Data:
	if robust:
		print("Tensorfying Pandas")
	else:
		print('')
	with tf.name_scope('input'):
		#Training Data
		train_matrix = training_set.as_matrix(columns=FEATURES)
		answer_matrix = training_set.as_matrix(columns=LABEL)
		
		if predictable:
			options_matrix = prediction_set.as_matrix(columns=FEATURES)
			predictions_matrix = prediction_set.as_matrix(columns=LABEL)

		#Input and outputs
		x = tf.placeholder(tf.float32, [None, 9], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')


	#Developing Neural Network:
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def variable_summaries(var):
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	def short_variable_summaries(var):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)

	def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
		with tf.name_scope(layer_name):
			with tf.name_scope('weights'):
				weights = weight_variable([input_dim, output_dim])
				variable_summaries(weights)
				variableWeights(weights)

			with tf.name_scope('biases'):
				biases = bias_variable([output_dim])
				variable_summaries(biases)
			with tf.name_scope('Wx_plus_b'):
				preactivate = tf.matmul(input_tensor, weights) + biases
				tf.summary.histogram('pre_activations', preactivate)

			activations = act(preactivate, name='activations')
			tf.summary.histogram('activations', activations)
			return activations
	
	def variableWeights(weights):
		inputMeans = 1
		with tf.name_scope('plantingDepth'):
			plantingDepthWeights = tf.gather(weights, 0)
			short_variable_summaries(plantingDepthWeights)
			if inputMeans:
				plantingDepthInputs = tf.gather(x, 0)
				short_variable_summaries(plantingDepthInputs)
		with tf.name_scope('soilAcidity'):
			soilAcidityWeights = tf.gather(weights, 1)
			short_variable_summaries(soilAcidityWeights)
			if inputMeans:
				SoilAcidityInputs = tf.gather(x, 1)
				short_variable_summaries(SoilAcidityInputs)
		with tf.name_scope('waterVapor'):
			waterVaporWeights = tf.gather(weights, 2)
			short_variable_summaries(waterVaporWeights)
			if inputMeans:
				waterVaporInputs = tf.gather(x, 2)
				short_variable_summaries(waterVaporInputs)
		with tf.name_scope('liquidWater'):
			liquidWaterWeights = tf.gather(weights, 3)
			short_variable_summaries(liquidWaterWeights)
			if inputMeans:
				liquidWaterInputs = tf.gather(x, 3)
				short_variable_summaries(liquidWaterInputs)
		with tf.name_scope('CO2'):
			COWeights = tf.gather(weights, 4)
			short_variable_summaries(COWeights)
			if inputMeans:
				COInputs = tf.gather(x, 4)
				short_variable_summaries(COInputs)
		with tf.name_scope('heatPad'):
			heatPadWeights = tf.gather(weights, 5)
			short_variable_summaries(heatPadWeights)
			if inputMeans:
				heatPadInputs = tf.gather(x, 5)
				short_variable_summaries(heatPadInputs)
		with tf.name_scope('redLight'):
			redLightWeights = tf.gather(weights, 6)
			short_variable_summaries(redLightWeights)
			if inputMeans:
				redLightInputs = tf.gather(x, 6)
				short_variable_summaries(redLightInputs)
		with tf.name_scope('blueLight'):
			blueLightWeights = tf.gather(weights, 7)
			short_variable_summaries(blueLightWeights)
			if inputMeans:
				blueLightInputs = tf.gather(x, 7)
				short_variable_summaries(blueLightInputs)
		with tf.name_scope('uvLight'):
			uvLightWeights = tf.gather(weights, 8)
			short_variable_summaries(uvLightWeights)
			if inputMeans:
				uvLightInputs = tf.gather(x, 8)
				short_variable_summaries(uvLightInputs)

	#Initialize layers for training (2 Layers)
	if robust:
		print("Creating neural network (layer1)")
	hidden1 = nn_layer(x, 9, 500, 'layer1')

	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		dropped = tf.nn.dropout(hidden1, keep_prob)

	if robust:
		print("Creating neural network (layer2)")
	y = nn_layer(dropped, 500, 1, 'layer2', act=tf.identity)

	'''
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		dropped = tf.nn.dropout(hidden1, keep_prob)

	if robust:
		print("Creating neural network (layer2)")
	hidden2 = nn_layer(dropped, 500, 500, 'layer2')

	with tf.name_scope('dropout2'):
		keep_prob = tf.placeholder(tf.float32)
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		dropped2 = tf.nn.dropout(hidden2, keep_prob)

	if robust:
		print("Creating neural network (layer2)")
	y = nn_layer(dropped2, 500, 1, 'layer2', act=tf.identity)
	'''

	#Analysis of Training
	with tf.name_scope('loss'):
		
		totalLoss = tf.reduce_sum(tf.abs(tf.subtract(y_,y)))
		absLoss = tf.reduce_mean(tf.abs(y_- y))
		tf.summary.scalar('avg_Loss', absLoss)
		allPredict = tf.reduce_mean(tf.abs(y_))
		#loss = absLoss
		lossDifference = tf.losses.absolute_difference(y_,y)
		loss = totalLoss
		tf.summary.scalar('sum_loss', loss)

	#Minimize loss using AdamOptimizer
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

	with tf.name_scope('accuracy'):
		#with tf.name_scope('correct_prediction'):
			#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		#with tf.name_scope('accuracy'):
		#accuracy = 100*absLoss/allPredict   #tf.reduce_mean(tf.abs(tf.divide(y_ - y, y_)))#*tf.constant(100.0))
		accuracy = loss
		tf.summary.scalar('accuracy', accuracy)

	realoutputs = y_
	outputs = y


	#Visualize the training info
	if robust:
		print("Creating directory for summaries\n")
	# Merge all the summaries and write them out to
	tensorboardDir = directory + 'TensorBoard/'

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(tensorboardDir + '/train', sess.graph)

	'''***** Use in console to open TensorBoard: 
	tensorboard --logdir=/Users/shanson/Desktop/Python/AI_Data/tensorboard --proxy=http://proxy-us.intel.com:911 
	*****'''

	def feed_dict(train):
		"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
		if train or FLAGS.fake_data:
			#print("Gathering train data")
			sess = tf.InteractiveSession()
			xs, ys = next_batch(batchSize, train_matrix, answer_matrix)
			k = FLAGS.dropout
		else:
			sess = tf.InteractiveSession()
			if predictable:
				if robust:
					print("Beginning prediction analysis")
				xs, ys = org_batch(options_matrix, predictions_matrix)		#Non shuffled batch
			else:
				xs, ys = next_batch(batchSize, train_matrix, answer_matrix)
			k = 1.0
		return {x: xs, y_: ys, keep_prob: k}

	print("Training:")

	sess.run(tf.global_variables_initializer())

	returnable = False

	for i in range(numCycles):
		#Predictions
		if i == numCycles-1 and predictable:
			returnable = ['predictions']
			print('\nPredicting outcomes')
			realouty, preouty = sess.run([realoutputs, outputs], feed_dict=feed_dict(False))
			print('\n' + str(len(preouty)) + ' predictions made\n')
			ab = 0
			if robust:
				for real in realouty:
					if ab % 25 == 0:
						absAdd = abs(real - preouty[ab])
						if robust:
							if absAdd > 0:
								print('Real: ' + str(real) + ', Predict: ' + str(preouty[ab]) + '. Difference = ' + str(absAdd))
					ab = ab + 1
			
				print(realouty)
				inwait = input('Realouty above. Press enter for predictouty below.')
				print(preouty)
				inwait = input('Combining lists')
			
			combined = []
			indexed = 0
			for real in realouty:
				row = []
				row.append(float(real))
				row.append(float(preouty[indexed]))
				combined.append(row)
				indexed = indexed + 1
			#flat = [x for sublist in outy for x in sublist]		#Flattend 2D numpy array to 1D list
			if standard:
				for row in combined:
					print(row)
				print('Combined list above of plant growth and predictions')
				inwait = input('Press enter to proceed.')
			returnable = combined

		#Regular Training
		if i % 10 == 0:				# Record summaries and test-set accuracy
			summary, lost = sess.run([merged, accuracy], feed_dict=feed_dict(True))
			#print('\n' + str(len(lost)) + ' losses?\n')
			print('Total Loss ' + str(i) +': ' + str(lost) + ', Average margin of error: ' + str(lost/batchSize))

			'''
			realout, preout = sess.run([realoutputs, outputs], feed_dict=feed_dict(True))
			
			ab = 0
			for real in realout:
				if ab % 25 == 0:
					absAdd = abs(real - preout[ab])
					print('Real: ' + str(real) + ', Predict: ' + str(preout[ab]) + '. Difference = ' + str(absAdd))
				ab = ab + 1
			'''
			
		else:						# Record train set summaries, and train
			if i % 100 == 99:		# Record execution stats
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True), 
							options=run_options, run_metadata=run_metadata)
				train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
				train_writer.add_summary(summary, i)
				print('Adding run metadata for', i)
			else:					# Record a summary
				summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
				train_writer.add_summary(summary, i)
	train_writer.close()
	'''
	saver = tf.train.Saver()
	sess.saver.restore(sess, "/tmp/model.ckpt")
	'''
	return returnable



# Section 5: Simulated 
def simulatePlantGrowth(plantList, simDataDir, simDataFile):
	print('Simulating Plant Growth')
	num = 0
	allReal = []
	for row in plantList:
		realSimData = []
		#Level 0
		#print(row)
		PlantingDepth, SoilAcidity, WaterVapor, LiquidWater = float(row[0]), float(row[1]), float(row[2]), float(row[3])
		CO2, HeatPad, RedLight, BlueLight, UVLight = float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])
		#Level 1
		RealSoilTemp = HeatPad*0.85 + UVLight*0.1 - LiquidWater*0.1
		Luminosity = UVLight*0.6 + RedLight*0.25 + BlueLight*0.15
		AirCO2 = (CO2*1.0 - WaterVapor*0.6)/100
		#Level 2
		AirTemp = RealSoilTemp*0.05 + UVLight*1.35*(1+AirCO2) - WaterVapor*0.33
		SoilHumidity = WaterVapor*0.05 + LiquidWater*0.95 - 0.05*abs(SoilAcidity-7) - RealSoilTemp*0.25
		#Level 3
		StemTemp = AirTemp*0.9 + RedLight*0.08 + BlueLight*0.02
		RootTemp = RealSoilTemp*0.9 + (AirTemp - PlantingDepth*10)*0.1 - SoilHumidity*0.2
		#Level 4
		RootGrowth = (6/7)*(70-abs(70 - SoilHumidity)) + (4/7)*(70-abs(70 - RootTemp))				#50%
		StemGrowth = RedLight*0.1 + BlueLight*0.15 + UVLight*0.05 + AirCO2*5 + StemTemp*0.2 + SoilHumidity*0.3	#50%
		#Level 5
		PlantingGrowth = (RootGrowth + StemGrowth)/2
		if robust:
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nSimulating plant ' + str(num))
			print('RealSoilTemp = ' + str(RealSoilTemp) + ', Luminosity = ' + str(Luminosity) + ', AirCO2 = ' + str(AirCO2))
			print('AirTemp = ' + str(AirTemp) + ', SoilHumidity = ' + str(SoilHumidity))
			print('StemTemp = ' + str(StemTemp) + ', RootTemp = ' + str(RootTemp))
			print('RootGrowth = ' + str(RootGrowth) + ', StemGrowth = ' + str(StemGrowth))
			print('\nReal Plant Growth = ' + str(PlantingGrowth))
			enter = input('Press enter to proceed.\n')
			os.system('cls')
		realSimData = row[:len(row)-1]
		realSimData.append(PlantingGrowth)
		allReal.append(realSimData)
		num = num + 1
	
	print()
	print(COLUMNS)
	for each in allReal:
		simplified = ['%.6f' % elem for elem in each]
		print(str(simplified))
	#Write data to Simulated Data csv
	wait = input('Press enter to proceed.')

	olderData = []
	existed = False
	if os.path.exists(simDataDir + simDataFile):
		with open(simDataDir + simDataFile) as csvfile:		#newline=''
			reader = csv.reader(csvfile)
			for row in reader:
				olderData.append(row)
		existed = True
	else:
		olderData.append(COLUMNS)
		del olderData[1]

	if len(olderData) == 0:
		olderData.append(COLUMNS)
	print(olderData)

	with open(simDataDir + simDataFile,'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',', quotechar='|', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
		if existed:
			writer.writerows(olderData)
		writer.writerows(allReal)
	csvfile.close()

def stichPredictions(directIn, fileIn, direct, preFile, predictions):
	if os.path.exists(directIn + fileIn):
		#print('File exists')
		comparedTable = []
		with open(directIn + fileIn) as csvfile:		#newline=''
			print('opened')
			reader = csv.reader(csvfile)
			x = 0
			titlesDone = False
			found = False
			problems = 0
			for row in reader:
				#print('test')
				if titlesDone:
					found = False
					for each in predictions:
						checkOnFile = "%.3f" % float(row[9]) 
						checkOnTrain = "%.3f" % float(each[0])
						if checkOnFile == checkOnTrain: 			#Stich predictions by finding it's associated row using sim growth as index
							row.append(each[1]) 	#Actual prediction
							#print('OH YEAH MAN')
							found = True
							print('RealOri: ' + str(checkOnFile) + ', RealPre: ' + str(checkOnTrain) + ', Prediction: ' + str(each[1]))
							break
						#bigOlWait = input('Waiting.')
					x = x + 1
				else:
					found = True
					row.append('Predictions')
					titlesDone = True
					
				comparedTable.append(row)

				#Find incomplete data (mainly due to rounding)
				#Reduce the sig figs to find the correllating prediction
				if len(row) < 11:
					if robust:
						print('Problematic row:')
						print(row)
						waitForIt = input('Enter to proceed')
					for each in predictions:
						checkOnFile = "%.2f" % float(row[9])
						checkOnTrain = "%.2f" % float(each[0])
						if checkOnFile == checkOnTrain:
							row.append(each[1])
							if robust:
								print(row)
								waitForFix = input('Found and fixed error.')
							found = True
							break

				#A Final Check
				if len(row) < 11:
					print('Problematic row:')
					print(row)

				if not found:
					problems = problems + 1
			if problems > 0:
				print('Could not find ' + str(problems) + ' lines to stich predictions to.')
			else:
				print('No errors thrown, all predictions mapped to input sets')
		csvfile.close()

		with open(directory + compareFile, 'w') as csvfile:		#newline=''
			writer = csv.writer(csvfile, delimiter=',', quotechar='|', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
			writer.writerows(comparedTable)
		csvfile.close()


	else:
		print('Predictions could not be written to comparison file!')
	wait = input('Press enter to end program.')

def inputSelection(numPlants, helpfulProb, trainCycles, batchSize, numChoices, 
		trainFromDir = directory, trainFromFile = originalFile, simulate = True):
	print('Creating Suggestions')
	suggested = getAllSuggestions(numPlants, helpfulProb, directory, originalFile)
	if suggested != False:
		numChoices, options = createChoices(suggested, directory, potentialFile)
		if robust:
			print('Excel written with ' + str(numChoices) + ' potential input choices!')
			wait = input('Press enter to proceed.')

	predictions = train(trainFromDir, trainFromFile, trainCycles, batchSize, numChoices, directory, potentialFile)

	if type(predictions) is not bool:
		if robust:
			print('Adding predictions to excel')
		print('Training Completed with Predictions!')

		row = 1
		
		#print(predictions)

		predictionTitle = 'Predictions'
		options[0].append(predictionTitle)
		for each in predictions:
			#print(str(options[row][9]))
			#if each[0] == options[row][9]:
			print(' Prediction: ' + str(each[1]))
			options[row].append(each[1])
			row = row + 1

		#Now to sort by newly added predictions to get best options to chose from:
		index = 0
		for each in options[0]:
			if each == predictionTitle:
				break
			else:
				index = index + 1
		#print('Type: ' + str(type(options)) + ' Index: ' + str(index))

		#Splice for sorting
		noTitles = options[1:][:]				#Just data
		sortedOptions = []
		sortedOptions.append(options[0])		#Just titles

		sortedChoices = sorted(noTitles, key=lambda x:x[index], reverse = True)

		for each in sortedChoices:
			sortedOptions.append(each)
		#print('Titles:\n' + str(sortedOptions[0]) + '\nLowest Data:\n' + str(sortedOptions[1]))

		numOptions = 10

		#Write to potentialFile
		with open(directory + potentialFile, 'w') as csvfile:		#newline=''
			writer = csv.writer(csvfile, delimiter=',', quotechar='|', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
			writer.writerows(sortedOptions)
		csvfile.close()
		if robust:
			print('Predictions added to ' + potentialFile)
		wait = input('Press Enter to Proceed')
		os.system('cls')
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		if standard:
			print('How many plants do you plan to sow?')
			numSow = input('Enter amount: ')
			numSow = int(numSow)
		else:
			print('Choosing ' + str(numPlants) + ' to sow.')
			numSow = numPlants
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		print('Would you like the computer to select these plants?')
		coCh = input('Enter yes (y) or no (n): ')
		compy = False
		if coCh == 'yes' or coCh == 'y' or coCh == 'Y' or coCh == 'Yes':
			compy = True
		else:
			compy = False
		if standard and not compy:
			print('How many prediction options would you like to choose from?')
			numOptions = input('Enter amount: ')
			numOptions = int(numOptions)
		else:
			print('Collecting ' + str(numOptions) + ' options to choose from.')
			numOptions = numOptions

		mutableChoices = sortedChoices
		sowPlants = []
		while numSow > 0:
			os.system('cls')
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			print('Top ' + str(numOptions) + ' options to choose from.')
			print(sortedOptions[0])		#Titles
			index = 0
			picked = []
			for each in mutableChoices[:numOptions][:]:
				simplified = ['%.6f' % elem for elem in each]
				print(str(index) + ') ' + str(simplified))
				picked.append(simplified)
				index = index + 1

			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			if standard and not compy:
				print('Please choose a plant from above to use.')
				chosenIndex = input('Choose numeric for choice: ')
				chosenIndex = int(chosenIndex)
			else:
				print('Always choosing best plant.')
				chosenIndex = 0
			sowPlants.append(mutableChoices[chosenIndex])
			del mutableChoices[chosenIndex]
			numSow = numSow - 1
		os.system('cls')
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		print('Here are your chosen plants you wish to sow:')
		print(sortedOptions[0])
		index = 0
		for each in sowPlants:
			simplified = ['%.6f' % elem for elem in each]
			print(str(index) + ') ' + str(simplified))
			index = index + 1
		print('')
		print('Run simulation through bot?')
		simit = input('Enter y or n: ')
		if simit == 'y':
			simulate = False
		else:
			simulate = True

		userIn = input('Press enter to proceed with simulation.')


		os.system('cls')
		if simulate:
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			simulatePlantGrowth(sowPlants, directory, simFile)
		else:		#Just create input sets
			print()
			print(COLUMNS)
			for each in sowPlants:
				simplified = ['%.6f' % elem for elem in each]
				print(str(simplified))
			#Write data to Simulated Data csv
			wait = input('Press enter to proceed.')

			olderData = []
			existed = False
			
			print('Fresh input sets?')
			refresh = input('Press y to refresh input sets or n to retain past input sets: ')
			if refresh == 'y':
				refreshFile(directory, inputSetsFile)

			titles = []
			for each in COLUMNS:
				if each == 'PlantGrowth':
					titles.append('Predictions')
				else:
					titles.append(each)	

			if os.path.exists(directory + inputSetsFile):
				with open(directory + inputSetsFile) as csvfile:		#newline=''
					reader = csv.reader(csvfile)
					for row in reader:
						olderData.append(row)
				existed = True
			else:
				olderData.append(titles)
				del olderData[1]

			if len(olderData) == 0:
				olderData.append(titles)
			elif len(olderData[0]) == 0:
				olderData[0] = titles 
			print(olderData)

			allReal = []
			#print(sowPlants)
			for each in sowPlants:
				allReal.append(each)
			#print(allReal)
					
			with open(directory + inputSetsFile,'w') as csvfile:
				writer = csv.writer(csvfile, delimiter=',', quotechar='|', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
				if existed:
					writer.writerows(olderData)
				writer.writerows(allReal)
			csvfile.close()


			print('Input sets file writen and ready to be implemented on the automated system.')
			sync = input('Press enter to exit.')

# Section 00: Main / Menu ----------------------------------------------------------------------

def main(_):
	#Run Mode
	os.system('cls')
	mode = 'quick' 				#quick = no userinput, regular = some userinput, robust = constant user input

	print('Select Run Mode\n\n(quick) Quick Mode - Fully automatic with no user input')
	print('(standard) Standard Mode - User input when user input is expected')
	print('(robust) Robust Mode - User input at every step to see full process')
	mode = input('\nPlease select a mode. (Write name in parentheses): ')

	refreshFile(directory, potentialFile)
	os.system('cls')

	global standard
	global robust

	if mode == 'quick':
		print('Quick Mode')
		standard, robust = False, False
	elif mode == 'robust':
		print('Robust Mode')
		standard, robust = True, True
	else:
		print('Standard Mode')
		standard, robust = True, False

	os.system('cls')

	#Setable Constants
	numPlants = 3				#For quick mode
	numOptions = 10				#For quick mode
	helpfulProb = 50
	createSuggestions = True

	#Initialize used variables/lists
	numChoices = 0
	options = []
	predictions = []

	exit = False
	while not exit:
		print('Main Menu')
		print('0) Refresh (delete & recreate) a file.')
		print('1) Train AI to test understanding.')
		print('2) Create Input Selection')
		print('3) Create Predictions for CSV Input Sets')
		print('\nQuit (q)')
		userInput = input('Enter index: ')
		if userInput == 'quit' or userInput == 'Quit' or userInput == 'q' or userInput == 'Q':
			break
		else:
			userInput = int(userInput)
		if type(userInput) is int and 0 <= userInput and userInput <= 3:
			#Refresh a File
			if userInput == 0:
				print('Refresh (delete & recreate) a File')
				filenames = listdir(directory)
				csvFiles = [filename for filename in filenames if filename.endswith(".csv")]
				index = 0
				for file in csvFiles:
					print(str(index) + ') ' + str(file))
					index = index + 1
				print('B) Back')
				userInput = input('Enter index: ')
				if userInput == 'b' or userInput == 'B' or userInput == 'Back' or userInput == 'back':
					print('Returning to main menu')
				else:
					userInput = int(userInput)
					refreshFile(directory, csvFiles[userInput])
			#Train AI
			elif userInput == 1:
				print('Train AI to Test Understanding')
				print('Choose a file to train from:')
				filenames = listdir(directory)
				csvFiles = [filename for filename in filenames if filename.endswith(".csv")]
				index = 0
				for file in csvFiles:
					print(str(index) + ') ' + str(file))
					index = index + 1
				print('B) Back')
				userInput = input('Enter index: ')
				if userInput == 'b' or userInput == 'B' or userInput == 'Back' or userInput == 'back':
					print('Returning to main menu')
				else:
					userInput = int(userInput)
					trainFile = csvFiles[userInput]
					print('How many training cycles?')
					trainCycles = int(input('# Cycles: '))
					print('What is the size of training batches?')
					trainBatch = int(input('Batch size: '))
					train(directory, trainFile, trainCycles, trainBatch)
					print('Training complete')
					closeTrain = input('View tensorboard or see data above, press enter to end session and return to main menu.')
					#exit = True

			elif userInput == 2:
				print('Create Input Selection')
				print('How many input choices?')
				numPlants = int(input('# input choices: '))
				print('How many training cycles?')
				trainCycles = int(input('# Cycles: '))
				print('What is the size of training batches?')
				trainBatch = int(input('Batch size: '))
				print('What is the percentage of best input values to use?')
				helpfulProb = int(input('Input percentage (0 to 100): '))
				print('Choose a file to train from:')
				filenames = listdir(directory)
				csvFiles = [filename for filename in filenames if filename.endswith(".csv")]
				index = 0
				for file in csvFiles:
					print(str(index) + ') ' + str(file))
					index = index + 1
				print('B) Back')
				userInput = input('Enter index: ')

				if userInput == 'b' or userInput == 'B' or userInput == 'Back' or userInput == 'back':
					print('Returning to main menu')
				else:
					goIn = input('Press enter to proceed')
					simulate = True
					inputSelection(numPlants, helpfulProb, trainCycles, trainBatch, numChoices, directory, csvFiles[int(userInput)], simulate)
				exit = True

			elif userInput == 3:
				os.system('cls')
				print('Create Prediction Set')
				print('')
				print('Select training CSV')
				filenames = listdir(directory)
				csvFiles = [filename for filename in filenames if filename.endswith(".csv")]
				index = 0
				for file in csvFiles:
					print(str(index) + ') ' + str(file))
					index = index + 1
				trainIn = input('Enter the index above: ')
				trainFrom = csvFiles[int(trainIn)]
				#print(trainFrom)

				print('')
				print('Select prediction CSV')
				filenames = listdir(directory)
				csvFiles = [filename for filename in filenames if filename.endswith(".csv")]
				index = 0
				for file in csvFiles:
					print(str(index) + ') ' + str(file))
					index = index + 1

				predictIn = input('Enter the index above: ')
				predictFrom = csvFiles[int(predictIn)]

				print('')
				print('How many cycles of training?')
				trainItUp = int(input('Enter integer: '))

				predictions = train(directory, trainFrom, trainItUp, 250, 500, directory, predictFrom)


				wait = input('Press enter to proceed.')
				refreshFile(directory, compareFile)
				comparedTable = []

				totalLoss = 0
				quantity = 0
				for each in predictions:
					totalLoss = totalLoss + abs(each[0] - each[1])
					quantity = quantity + 1

				avgLoss = totalLoss/quantity

				print('Average Loss: ' + str(avgLoss))

				
				if os.path.exists(directory + predictFrom):
					#print('File exists')
					comparedTable = []
					with open(directory + predictFrom) as csvfile:		#newline=''
						#print('opened')
						reader = csv.reader(csvfile)
						x = 0
						titlesDone = False
						found = False
						problems = 0
						for row in reader:
							if x > 0:
								print('Predictions: ' + str(predictions[x-1][1]) + ', Actual: ' + str(row[9]))
								row.append(predictions[x-1][1]) 	#Actual prediction
							x = x + 1
					csvfile.close()

					with open(directory + compareFile, 'w') as csvfile:		#newline=''
						writer = csv.writer(csvfile, delimiter=',', quotechar='|', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
						writer.writerows(comparedTable)
					csvfile.close()

				else:
					print('Predictions could not be written to comparison file!')
				wait = input('Press enter to end program.')
				
				#Get Real Data:
				'''
				if os.path.exists(directory + compareFile):
					print('Stiching')
					stichPredictions(directory, compareFile, directory, compareFile, predictions)
				break
				'''

		else:
			print('The input you entered is not an option.\nPlease enter an int between 0 and 2 or enter \'quit\'.')
			wait = input('Press enter to proceed.')
		os.system('cls')
		print('Entering Main Menu')
		time.sleep(1)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
		default=False, help='If true, uses fake data for unit testing.')
	parser.add_argument('--max_steps', type=int, default=200,
		help='Number of steps to run trainer.')
	parser.add_argument('--learning_rate', type=float, default=0.001,
		help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default=0.9,
		help='Keep probability for training dropout.')
	parser.add_argument('--data_dir', type=str, 
		default='/Users/shanson/Desktop/Python/AI_Data/TrainTemp',
		help='Directory for storing input data')
	parser.add_argument('--log_dir', type=str,
		default='/tmp/tensorflow/mnist/logs/soiltemp_with_summaries',
		help='Summaries log directory')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
	#tf.app.run()

main()

'''
mount /dev/sda1 flash
cd flash
python SA.py
'''