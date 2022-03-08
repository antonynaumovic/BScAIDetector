from imageai.Detection.Custom import CustomObjectDetection


detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(r"E:\Computing Project BSc\AI Files\roadobjects\models\detection_model-ex-005--loss-0018.929.h5")
detector.setJsonPath(r"E:\Computing Project BSc\AI Files\roadobjects\json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=r"E:\Computing Project BSc\AI Files\roadobjects\validation\images\image0111.png", output_image_path=r"E:\Computing Project BSc\AI Files\Evaluate\road52Eval.png")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])