import pathlib
import shutil

def main():
    root = pathlib.Path(__file__).parent / 'runs' / 'detect'
    for train_dir in root.iterdir():
        if not train_dir.is_dir() or 'train' not in train_dir.name:
            continue
        res_file = train_dir / 'results.csv'
        if not res_file.exists():
            print(train_dir)
            shutil.rmtree(train_dir)
            
            
if __name__ == "__main__":
    main()