from nanpy import (ArduinoApi, SerialManager)
from time import sleep

###  BEGIN GLOBAL ARDUINO DEFINITIONS  ###

# define connections and arduino object variables

try:
    connection = SerialManager()
    a = ArduinoApi(connection = connection)
except:
    print "Failed to connect to Arduino"

# define logic control output pins
in1 = 9
in2 = 8
in3 = 7
in4 = 6

# define channel enable output pins
ENA = 10
ENB = 5

### END GLOBAL  ARDUINO PIN DEFINITIONS  ###

### BEGIN ARDUINO HELPER FUNCTIONS

def mForward():
    
    a.digitalWrite(ENA, a.HIGH)
    a.digitalWrite(ENB, a.HIGH)
    
    a.digitalWrite(in1, a.LOW)
    a.digitalWrite(in2, a.HIGH)
    a.digitalWrite(in3, a.LOW)
    a.digitalWrite(in4, a.HIGH)
    
    print "Car moving forward\n"
    
    return

def mBack():

    a.digitalWrite(ENA, a.HIGH)
    a.digitalWrite(ENB, a.HIGH)
    
    a.digitalWrite(in1, a.HIGH)
    a.digitalWrite(in2, a.LOW)
    a.digitalWrite(in3, a.HIGH)
    a.digitalWrite(in4, a.LOW)

    print "Car moving backward\n"
    
    return

def ArduinoSetup():

    a.pinMode(in1, a.OUTPUT)
    a.pinMode(in2, a.OUTPUT)
    a.pinMode(in3, a.OUTPUT)
    a.pinMode(in4, a.OUTPUT)
    a.pinMode(ENA, a.OUTPUT)
    a.pinMode(ENB, a.OUTPUT)
    
    print "Arduino pin setup complete\n"

    return


###  END ARDUINO HELPER FUNCTIONS  ###


def stopCar():
	
	return;
def moveCarForward(amount):

	return;

def takePicture():
	stopCar();
	# takePicture
	picture = []; # fix this crap
	return picture;


def classifyWeedOrCrop(pic):

	# ryan, dude, really?
	return

def classifyCrop(pic):

	# ankeet get this shite done mate
	return

def process():
	pic = takePicture();
	result = classifyWeedOrCrop(pic);
	if (result == "weed"):
		print "weed";
	else:
		print "crop"
		cropType = classifyCrop(pic);
		print cropType;


def main():
    
    ArduinoSetup()

	# todo: run for ~20 seconds
	amount = 10; # inches
	while True:
		moveCarForward(amount);
		process();

	print("we are done")

main();


