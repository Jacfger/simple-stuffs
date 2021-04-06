import torch
from torch import optim
import tqdm

from fol import TransE_Tnorm, parse_foq_formula

if __name__ == "__main__":
    mock_dataset = ("[7,8,9]([1,2,2]({1,1,3})&[3,3,4]({6,5,6}))", [[2], [4], [6]])
    X, Y = mock_dataset
    foq_instance = parse_foq_formula(X)
    print(foq_instance.ground_formula)


    model = TransE_Tnorm()
    opt = optim.SGD(model.parameters(), lr=1e-3)

    with tqdm.trange(10000) as t:
        for i in t:
            opt.zero_grad()
            pred = foq_instance.embedding_estimation(estimator=model)
            loss = model.criterion(pred, Y)
            loss.backward()
            opt.step()
            t.set_postfix({'loss': loss.item()})




