from api import getFlooding


if __name__ == "__main__":
    data = getFlooding()
    data.to_csv("wl.csv", index=False)
