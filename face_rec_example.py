import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6 # Higher the value greater the false positives(it is you in the image but it won't recognize you) and lower the value greater the true negatives(it will classify others image also as being you)
FRAME_THICKNESS = 3 # Bounding box size around the face
FONT_THICKNESS = 2
MODEL = "cnn" # other models "hog"

print("loading known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)

print("processing unknown faces")

for filename in os.listdir(UNKNOWN_FACES_DIR):
	print(filename)
	image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
	# dectecting all the faces in unknown faces image and finding the locations(basically doing face dectection)
	locations = face_recognition.face_locations(image, model=MODEL)
	encodings = face_recognition.face_encodings(image, locations)
	# Open the image in cv2 acceptable format
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	for face_encoding, face_location in zip(encodings, locations):
		results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
		match = None
		if True in results:
			match = known_names[results.index(True)]
			print(f"Match found: {match}")

			# For building box around the face
			top_left = (face_location[3], face_location[0])
			bottom_right = (face_location[1], face_location[2])

			# Color of the box
			color = [0, 255, 0] # (BGR)Green

			# Draw the rectangle
			cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

			# For building box to display the name
			top_left = (face_location[3], face_location[2])
			bottom_right = (face_location[1], face_location[2]+22)

			# Draw the rectangle where the name is displayed
			cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

			# Text to be displayed
			cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)

	cv2.imshow(filename, image)
	cv2.waitKey(10000)
	#cv2.destroyWindow(filename)




			