import streamlit as st
from fastai.vision.all import *


def extractBreed(filePath):
    species = ""
    path = Path(filePath)
    breed = path.stem
    parts = breed.split("_")
    breed = "_".join(parts[:-1])
    if breed.islower():
        species = "DOG"
    else:
        species = "CAT"

    result = f"{species} - {breed}"

    return result

catAndDogBreedModel = load_learner("CatAndDogBreed(2).pkl")

def predict(image):
    img = PILImage.create(image)
    resize = img.resize((224, 224))
    prediction = catAndDogBreedModel.predict(resize)
    breed_index = prediction[1].item()
    accuracy = prediction[2][breed_index]
    # print(prediction)
    # breedDict = dls.vocab
    # print(type(prediction[2]))
    # for i, prob in enumerate(prediction[2]):
    # print(breedDict[i], prob)

    if accuracy >= 0.99:
        return f"{prediction[0]}, {(accuracy * 100):.2f}%"
    else:
        return f"I'm not sure - I am {(accuracy * 100):.2f}% sure that it is a {prediction[0]}."

    #img.show(title=f"{prediction[0]}, {(accuracy * 100):.2f}%")

st.title("Crystal's Cat And Dog Breed Guesser")
st.text("Built by Crystal")

uploadedFile = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploadedFile is not None:
    st.image(uploadedFile, caption="Your Image!", use_column_width=True)

    prediction = predict(uploadedFile)
    st.write(prediction)

st.text("Built with Streamlit and Fastai.")