import csv, sys, time, os, datetime

import subprocess

def replacer(oldstring, add, index, nofail=False):

	if index < 0:
		return add + oldstring
	if index > len(oldstring):
		return oldstring + add

	newstring = oldstring[:index] + [str(add)] + oldstring[index + 1:]

	return newstring

#categories = 'Date,Time,TestNumber,PlantingDepth,SoilAcidity,WaterVapor,LiquidWater,CO2,HeatPad,RedLight,BlueLight,UVLight,PlantGrowth'
categories = 'Date,Time,TestNumber,Plant,UV,Light,Distance,Moisture,Temperature'
directory = 'AI_Data/PlantGrowth'
realData = '/RealData.csv'
simulatedData = '/SimulatedData.csv'
inputSets = '/InputSets.csv'

def clean(direct, file):
	#Remove the file to create new?
	if os.path.exists(direct + file):
		os.remove(direct + file)

	#If no file, create it and set category header
	if not os.path.exists(direct + file):
		with open(direct + file, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer.writerow([categories])

#Add data to the earliest open slot
def addDataInLine(category, value, direct, file):

	print('Add data in line for ' + category + ' (' + str(value) + ')')
	index = 0
	listCat = []
	x = 0
	startCut = 0
	while x < len(categories):
		if categories[x] == ',':
			listCat.append(categories[startCut:x])
			startCut = x + 1
		elif x == len(categories)-1:
			listCat.append(categories[startCut:x+1])
		x = x + 1


	if index < 0:
		index = 0

	listIndex = 0
	while listIndex < len(listCat):
		if category == listCat[listIndex]:
			break
		else:
			index = index + 1
			listIndex = listIndex + 1

	with open(direct + file) as csvfile:

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

				blankrow = []
				for each in listCat:
					blankrow.append('-') 
				blankrow[index] = str(value)
				blankrow[0] = str(curdate) 
				blankrow[1] = str(curtime)
				blankrow[2] = str(numTest)

				newRow = ''
				for each in blankrow:
					newRow = newRow + str(each) + ','
				print(newRow)
				rowlist.append(newRow)

		except csv.Error as e:
			sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))

	with open(direct + file, 'w') as csvfile:

		writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

		for numb in rowlist:
			writer.writerow([numb])

#Lock all tests, so that new tests can be added
def lockPastTests(direct, file):

	print('Locking past tests')
	rowlist = []
	with open(direct + file) as csvfile:

		reader = csv.reader(csvfile)
		try:
			for row in reader:  

				x = 0
				while x < len(row):
					if row[x] == '-':
						row[x] = '~'
					x = x + 1
		
				rowlist.append(','.join(row).replace('|', ''))
				print(rowlist)

		except csv.Error as e:
			sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))

	with open(direct + file, 'w') as csvfile:

		writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for numb in rowlist:
			writer.writerow([numb])

def addNewData(category, value):

	print('Adding new data')
	lockPastTests()
	addDataInLine(category, value)





#!/usr/bin/python

# Author: HARRY CHAND <hari.chand.balasubramaiam@intel.com>
# Copyright (c) 2014 Intel Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import time, sys, signal, atexit, mraa, thread, threading, os
import pyupm_grove as grove
import pyupm_guvas12d as upmUV
import pyupm_grovemoisture as upmMoisture
import pyupm_stepmotor as mylib
import pyupm_servo as servo
from threading import Thread
from multiprocessing import Process

# IO Def
myIRProximity = mraa.Aio(5)  				#GP2Y0A on Analog pin 5
temp = grove.GroveTemp(0) 					#grove temperature on A0 
myMoisture = upmMoisture.GroveMoisture(1) 	#Moisture sensor on A1
light = grove.GroveLight(2) 				#Light sensor on A2
myUVSensor = upmUV.GUVAS12D(3) 				#UV sensor on A3
stepperX = mylib.StepMotor(2, 3) 			#StepMotorY object on pins 2 (dir) and 3 (step)
stepperY = mylib.StepMotor(4, 5)			#StepMotorX object on pins 4 (dir) and 5 (step)
waterpump = mraa.Gpio(10) 					#Water pump's Relay on GPIO 10
waterpump.dir(mraa.DIR_OUT)
waterpump.write(0)
gServo = servo.ES08A(6)                		#Servo object using D6
gServo.setAngle(50)
switchY = mraa.Gpio(7)    					#SwitchY for GPIO 7
switchY.dir(mraa.DIR_IN)
switchX = mraa.Gpio(8)						#SwitchX for GPIO 8
switchX.dir(mraa.DIR_IN)
EnableStepper = mraa.Gpio(9)				#StepperMotor Enable on GPIO 9
EnableStepper.dir(mraa.DIR_OUT)
EnableStepper.write(0)
button = grove.GroveButton(0)  				#Digital Button on D0   -> ## button.value()

		
# Variable Def
AREF = 5.0
SAMPLES_PER_QUERY = 1024
flag = 1

# Defined all 5 Sensors
UVvalue = myUVSensor.value(AREF, SAMPLES_PER_QUERY) 				#Voltage value (higher means more UV)
Lightvalue = light.value()  										# in lux
Distancevalue = float(myIRProximity.read())*AREF/SAMPLES_PER_QUERY  #Distance in Voltage (higher mean closer)
Soilvalue = myMoisture.value() 										# 0-300 Dry, 300-600 Most, <600 Wet
Tempvalue = temp.value()  											# Celsius

# Defined Motors
stepperX.setSpeed(150)
stepperY.setSpeed(150)

## Exit handlers ##
# This function stops python from printing a stacktrace when you hit control-C
def SIGINTHandler(signum, frame):
	raise SystemExit

# This function lets you run code on exit, including functions from myUVSensor
def exitHandler():
	waterpump.write(0)
	EnableStepper.write(0)
	restart.terminate()
	Mx.terminate()
	My.terminate()
	print "Exiting"
	sys.exit(0)

# Register exit handlers
atexit.register(exitHandler)
signal.signal(signal.SIGINT, SIGINTHandler)
def init_MotorX():
	if (switchX.read()):
		stepperX.stepForward(3000)
		time.sleep(0.3)
	return
	
def init_MotorY():
	if (switchY.read()):
		stepperY.stepBackward(3000)
		time.sleep(0.3)
	return
	
def Restart_Program():
	while (button.value() == 1):
		EnableStepper.write(1)
		waterpump.write(0)
		EnableStepper.write(0)
		sensor.terminate()
		restart.terminate()
		Mx.terminate()
		My.terminate()
		os.execv(sys.executable, sys.executable + sys.argv) #os.execv(sys.executable, ['python'] + sys.argv)
	return
	
def initial():
	print "Reset to initial stages ..... "
	# Test input value for switch(s) and restart button
	# Test Stepper Motor (going to initial stages)
	EnableStepper.write(0)
	Mx.start()
	My.start()
	while (switchX.read() | switchY.read()):
		if (switchX.read()==0): Mx.terminate()
		if (switchY.read()==0): My.terminate()
	EnableStepper.write(1)
	# Turn OFF water pump relay
	waterpump.write(0)
	# Servo z-axis should be up
	gServo.setAngle(50)
	return

#Simulated Data:
simColumns = ["Plant", "PlantingDepth", "SoilAcidity", "WaterVapor", "LiquidWater", "CO2",
			"HeatPad", "RedLight", "BlueLight", "UVLight", "PlantGrowth"]

def simulateGrowth(inData, setDir, setFle):
	print('Simulating growth')
	robust = True
	realSimData = []
	#Level 0
	#print(row)
	PlantingDepth, SoilAcidity, WaterVapor, LiquidWater = float(inData[1]), float(inData[2]), float(inData[3]), float(inData[4])
	CO2, HeatPad, RedLight, BlueLight, UVLight = float(inData[5]), float(inData[6]), float(inData[7]), float(inData[8]), float(inData[9])
	#Level 1
	RealSoilTemp = HeatPad*0.85 + UVLight*0.1 - LiquidWater*0.1
	if RealSoilTemp < 0:
		RealSoilTemp = 0
	Luminosity = UVLight*0.6 + RedLight*0.25 + BlueLight*0.15
	AirCO2 = (CO2*1.0 - WaterVapor*0.6)/100
	if AirCO2 < 0:
		AirCO2 = 0
	#Level 2
	AirTemp = RealSoilTemp*0.05 + UVLight*1.35*(1+AirCO2) - WaterVapor*0.33
	if AirTemp < 0:
		AirTemp = 0
	SoilHumidity = WaterVapor*0.05 + LiquidWater*0.95 - 0.05*abs(SoilAcidity-7) - RealSoilTemp*0.25
	if SoilHumidity < 0:
		SoilHumidity = 0
	#Level 3
	StemTemp = AirTemp*0.9 + RedLight*0.08 + BlueLight*0.02
	RootTemp = RealSoilTemp*0.9 + (AirTemp - PlantingDepth*10)*0.1 - SoilHumidity*0.2
	if RootTemp < 0:
		RootTemp = 0
	#Level 4
	RootGrowth = (6/7)*(70-abs(70 - SoilHumidity)) + (4/7)*(70-abs(70 - RootTemp))				#50%
	if RootGrowth < 0:
		RootGrowth = 0
	StemGrowth = RedLight*0.1 + BlueLight*0.15 + UVLight*0.05 + AirCO2*5 + StemTemp*0.2 + SoilHumidity*0.3	#50%
	#Level 5
	PlantingGrowth = (RootGrowth + StemGrowth)/2
	if robust:
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nSimulating plant ' + inData[0])
		time.sleep(1)
		print('RealSoilTemp = ' + str(RealSoilTemp) + ', Luminosity = ' + str(Luminosity) + ', AirCO2 = ' + str(AirCO2))
		time.sleep(1)
		print('AirTemp = ' + str(AirTemp) + ', SoilHumidity = ' + str(SoilHumidity))
		time.sleep(1)
		print('StemTemp = ' + str(StemTemp) + ', RootTemp = ' + str(RootTemp))
		time.sleep(1)
		print('RootGrowth = ' + str(RootGrowth) + ', StemGrowth = ' + str(StemGrowth))
		time.sleep(1)
		print('\nReal Plant Growth = ' + str(PlantingGrowth))
		time.sleep(3)
	realSimData = inData[1:len(inData)-1]
	realSimData.append(PlantingGrowth)
	print('Real simulated Data inputs/outputs')
	print(realSimData)

	#Get Real Data:
	simmedData = []
	if os.path.exists(directory + simulatedData):
		with open(directory + simulatedData) as csvfile:		#newline=''
			reader = csv.reader(csvfile)
			titlesDone = False
			x = 0
			simmedData.append(simColumns)
			for row in reader:
				if x > 0:
					simmedData.append(row)
				x = x + 1
		csvfile.close()

		simmedData.append(realSimData)

		with open(directory + simulatedData, 'w') as csvfile:		#newline=''
			writer = csv.writer(csvfile, delimiter=',', quotechar='|', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
			writer.writerows(simmedData)
		csvfile.close()
	else:
		print('No file exists, cant simulate growth!')

	#allReal.append(realSimData)

	time.sleep(10)

#Progress Functions:
def progressPlants():
	#subprocess.Popen('clear')
	#print(chr(27)+"[2J")
	index = 0
	for each in currentGrowth[0]:
		if each == 'Progress':
		 	break
		index = index + 1

	#print('Progress Index ' + str(index))
	x = 0
	plantQ = 0
	if len(currentGrowth) > 1:
		titles = []
		for current in currentGrowth:
			if x > 0:
				current[index] = str(int(current[index]) + 5)
				if int(current[index]) >= 100:
					print('Finished plant ' + current[0] + ' at a final plant growth of ' + current[index-1])
					simulateGrowth(current, directory, simulatedData)
					currentGrowth.pop(plantQ)
				else:
					if x == 1:
						print(titles)
					print current
			else:
				titles = current
			plantQ = plantQ + 1
			x = x + 1
		print('')
	
def MoveToPot(pot):
	print "Moving to Pot %d " %(pot)
	pot = pot + 1
	posX = 200; posY = 200
	running = True
	progressPlants()
	if running:
		EnableStepper.write(0)
		if (pot == 1): stepperX.stepBackward(posX); stepperY.stepForward(posY); 
		elif (pot == 2): stepperX.stepBackward(posX+100); stepperY.stepForward(posY+100); 
		elif (pot == 3): stepperX.stepBackward(posX+200); stepperY.stepForward(posY+200); 
		elif (pot == 4): stepperX.stepBackward(posX+300); stepperY.stepForward(posY+300); 
		elif (pot == 5): stepperX.stepBackward(posX+400); stepperY.stepForward(posY+400); 
		elif (pot == 6): stepperX.stepBackward(posX+500); stepperY.stepForward(posY+500); 
		elif (pot == 7): stepperX.stepBackward(posX+600); stepperY.stepForward(posY+600); 
		elif (pot == 8): stepperX.stepBackward(posX+700); stepperY.stepForward(posY+700); 
		elif (pot == 9): stepperX.stepBackward(posX+800); stepperY.stepForward(posY+800); 
		else: print "Invalid operation for Pot Position";
		EnableStepper.write(1)
	else:
		time.sleep(1)

	print('Simulating sowing')
	if running:
		gServo.setAngle(100)
	time.sleep(1)
	progressPlants()
	time.sleep(1)
	progressPlants()
	time.sleep(1)
	progressPlants()
	if running:
		gServo.setAngle(50)
	#print('Done simulating')

	if running:
		enableStepper.write(0)
		if (pot == 1): stepperX.stepForward(posX); stepperY.stepBackward(posY); 
		elif (pot == 2): stepperX.stepForward(posX+100); stepperY.stepBackward(posY+100); 
		elif (pot == 3): stepperX.stepForward(posX+200); stepperY.stepBackward(posY+200); 
		elif (pot == 4): stepperX.stepForward(posX+300); stepperY.stepBackward(posY+300); 
		elif (pot == 5): stepperX.stepForward(posX+400); stepperY.stepBackward(posY+400); 
		elif (pot == 6): stepperX.stepForward(posX+500); stepperY.stepBackward(posY+500); 
		elif (pot == 7): stepperX.stepForward(posX+600); stepperY.stepBackward(posY+600); 
		elif (pot == 8): stepperX.stepForward(posX+700); stepperY.stepBackward(posY+700); 
		elif (pot == 9): stepperX.stepForward(posX+800); stepperY.stepBackward(posY+800); 
		else: print "Invalid operation for Pot Position";
		EnableStepper.write(1)
	else:
		time.sleep(1)

	return 
	
def addAllData(plantNumber, directory, file):
	print('Storing all data from plant ' + str(plantNumber))
	addDataInLine('Plant', plantNumber, directory, file)
	print "Test all the 5 SENSORS :"
	uv = UVvalue
	print "1. UV Sensor : 		%d V" % uv
	addDataInLine('UV', uv, directory, file)
	light = Lightvalue
	print "2. Light Sensor : 	%d Lux" % light
	addDataInLine('Light', light, directory, file)
	distance = Distancevalue
	print "3. Distance Sensor : 	%f V" % distance
	addDataInLine('Distance', distance, directory, file)
	moisture = Soilvalue
	print "4. Moisture Sensor : 	%d " % moisture
	addDataInLine('Moisture', moisture, directory, file)
	temp = Tempvalue
	print "5. Temperature Sensor :  %d Celsius" % temp
	addDataInLine('Temperature', temp, directory, file)

categories = 'Date,Time,TestNumber,Plant,UV,Light,Distance,Moisture,Temperature'

currentGrowth = []
#currentGrowth.append('RealGrowth')
simColumns.append('Progress')
currentGrowth.append(simColumns)
#print(currentGrowth)

currentlyMoving = False

def sowPlant(plantNumber, inData):
	print('Sowing plat at position ' + str(plantNumber))
	currentlyMoving = True
	betterList = []
	betterList.append(str(plantNumber))
	for each in inData:
		betterList.append(each)
	betterList.append(0)
	MoveToPot(int(plantNumber))
	currentGrowth.append(betterList)
	#time.sleep(5)
	currentlyMoving = False	


def getInputs(inDirect, inFile):
	#print('Its all a simulation man.')
	dataSet = []
	existed = False
	if os.path.exists(inDirect + inFile):
		existed = True
		with open(inDirect + inFile) as csvfile:		#newline=''
			reader = csv.reader(csvfile)
			for row in reader:
				dataSet.append(row)
	if existed:
		return dataSet
	else:
		return False
		

# Global Definition
restart = Process(target = Restart_Program) #Go into Initial Stages
Mx = Process(target = init_MotorX)
My = Process(target = init_MotorY)

if __name__ == '__main__':
	while (flag):
		clean(directory, realData)
		clean(directory, simulatedData)
		restart.start()
		# Add calling camera modules
		initial()

		continued = True

		while (continued):
			print('Getting Input Data')
			gotInputs = getInputs(directory, inputSets)
			#print(gotInputs)
			if isinstance(gotInputs, bool):
				filled = False
			else:
				filled = True

			progressPlants()

			while (filled):
				currentSpots = []
				x = 0
				for growth in currentGrowth:
					if x > 0:
						currentSpots.append(int(growth[0]))
					x = x + 1
				
				plant = 0
				noneOpen = False
				while plant < 9:
					openSpot = True
					for spot in currentSpots:
						if plant == spot:
							openSpot = False
					if openSpot:
						break
					plant = plant + 1
					if plant == 9:
						openSpot = False

				#Planting Functions
				if not currentlyMoving:
					plant = 0
					noneOpen = False
					while plant < 9:
						openSpot = True
						for spot in currentSpots:
							if plant == spot:
								openSpot = False
						if openSpot:
							break
						plant = plant + 1
						if plant == 9:
							openSpot = False

					if openSpot:
						simplified = []
						if len(gotInputs) > 1:
							#print(len(gotInputs))
							#print(gotInputs)
							for each in gotInputs[1]:
								simplified.append('%.6f' % float(each))
							sowPlant(plant, simplified)
							gotInputs.pop(1)
						else:
							filled = False
					else:
						print('No open spots')

				#print('Progress!')
				progressPlants()
				time.sleep(1)

			continued = True
			print('No new plants to sow but continuing growth')
			time.sleep(5)
			if len(currentGrowth) < 2:
				continued = False
				print('No new plants to sow and no growth')

		flag = 0
	
	init_MotorX()
	init_MotorY()
	#inital()
	del [light, temp, button, gServo]  
	restart.terminate()
	Mx.terminate()
	My.terminate()

# Information:
## Soil moisture Values (approximate):
# 0-300,   sensor in air or dry soil
# 300-600, sensor in humid soil
# 600+,    sensor in wet soil or submerged in water

## Infrared Proximity Sensor 
# The higher the voltage (closer to AREF) the closer the object is.
# NOTE: The measured voltage will probably not exceed 3.3 volts.
# Every second, print the averaged voltage value
# (averaged over 20 samples).
