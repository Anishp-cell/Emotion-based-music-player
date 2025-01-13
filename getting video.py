import cv2
# Initialize video capture
cap = cv2.VideoCapture(0)

# Print initial frame dimensions
print("Initial Frame Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Initial Frame Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set desired frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Print updated frame dimensions
print("Updated Frame Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Updated Frame Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Add text overlay with frame dimensions
        font = cv2.FONT_ITALIC
        text = f'Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}'


        # Display the frame
        cv2.imshow('frame', frame)

        # Break the loop if 'ESC' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print("Error: Failed to read frame.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
