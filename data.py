from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import cv2 
import joblib

ROOT_DIR = "../data/googleLandmarkRetrieval/"
TRAIN_DIR = "../data/googleLandmarkRetrieval/train/"
TEST_DIR = "../data/googleLandmarkRetrieval/test/"

def get_train_path(id):
    return f"{TRAIN_DIR}/{id[0]}/{id[1]}/{id[2]}/{id}.jpg"

    
def get_dataframes(CONFIG):
    df = pd.read_csv(f"{ROOT_DIR}/train.csv")
    
    le = LabelEncoder()
    df.landmark_id = le.fit_transform(df.landmark_id)
    joblib.dump(le, "label_encoder.pkl")

    df['file_path'] = df['id'].apply(get_train_path)
    df_train, df_test = train_test_split(df, test_size=0.4, stratify=df.landmark_id, shuffle=True, random_state=CONFIG['seed'])
    df_valid, df_test = train_test_split(df_test, test_size=0.5, shuffle=True, random_state=CONFIG['seed'])

    return df_train, df_valid, df_test

class LandmarkDataset(Dataset):
    def __init__(self, root_dir, df, transforms=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.file_names = df["file_path"].values
        self.df = df 
        self.labels = df['landmark_id'].values 
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
            img_path = self.file_names[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = self.labels[index]

            if self.transforms:
                img = self.transforms(image=img)["image"]
            
            return img, label 

def prepare_loaders(CONFIG):    
    df_train, df_valid, _ = get_dataframes(CONFIG)
    

    data_transforms = {
    "train": A.Compose([
        A.Resize(CONFIG["img_size"], CONFIG["img_size"]), 
        A.HorizontalFlip(p=.5), 
        A.CoarseDropout(p=.5), 
        A.Normalize(max_pixel_value=255.0, p=1.0), 
        ToTensorV2()
    ], p=1.), 

    "valid": A.Compose([
        A.Resize(CONFIG["img_size"], CONFIG["img_size"]), 
        A.Normalize(max_pixel_value=255.0, p=1.0), 
        ToTensorV2()
    ], p=1.)

}
    
    train_dataset = LandmarkDataset(TRAIN_DIR, df_train, transforms=data_transforms['train'])
    valid_dataset = LandmarkDataset(TRAIN_DIR, df_valid, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=4, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=4, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader