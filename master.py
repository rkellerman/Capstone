# import shit here

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

	# todo: run for ~20 seconds
	amount = 10; # inches
	while True:
		moveCarForward(amount);
		process();

	print("we are done")

main();


