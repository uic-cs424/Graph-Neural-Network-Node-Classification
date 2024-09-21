import torch


# Defining the training loop

def run_training(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test():
        model.eval()
        pred = model(data.x, data.edge_index).argmax(dim=-1)

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        return accs

    train_acc_list, val_acc_list, test_acc_list = [], [], []
    best_val_acc = final_test_acc = 0
    for epoch in range(1, epochs + 1):
        loss = train()
        train_acc, val_acc, tmp_test_acc = test()
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(tmp_test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    return train_acc_list, val_acc_list, test_acc_list
