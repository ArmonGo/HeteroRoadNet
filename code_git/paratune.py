from model import * 
from utilis import evaluate, train
import optuna
import schedulefree
# define the tuning model 

def objective_hetero(trial, train_loader, val_loader,node_types, edge_types, epochs, device, loss_f):
    dict_out_channels = trial.suggest_int("dict_out_channels", 4, 64, 4)
    pois_out_channels = trial.suggest_int("pois_out_channels", 4, 64, 4)
    pois_drop_out  =  trial.suggest_float("pois_drop_out", 0, 0.7)
    target_out_channels = trial.suggest_int("target_out_channels", 4, 64, 4)
    target_drop_out  =  trial.suggest_float("target_drop_out", 0, 0.7)
    road_out_channels = trial.suggest_int("road_out_channels", 4, 64, 4)
    road_drop_out  =  trial.suggest_float("road_drop_out", 0, 0.7)
    all_out_channels = trial.suggest_int("all_out_channels", 4, 64, 4)
    all_drop_out  =  trial.suggest_float("all_drop_out", 0, 0.7)
    # Generate the model.
    model = HeteroGNN(  node_types, edge_types, 
                        dict_out_channels,
                        pois_out_channels,  pois_drop_out, 
                        target_out_channels, target_drop_out,
                        road_out_channels, road_drop_out,
                        all_out_channels, all_drop_out).to(device)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr)
    for epoch in range(epochs):
        tr_loss = train(model, device, optimizer, dataloader = train_loader, data = None, loss_f = loss_f)
        val_loss, _, _ = evaluate(model, device, optimizer, dataloader = val_loader, data = None, loss_f = loss_f)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return val_loss

def objective_homo(trial, tr_g, val_g, epochs, device, loss_f):
    out_channels_layer1 = trial.suggest_int("out_channels_layer1", 4, 64, 4)
    out_channels_layer2 = trial.suggest_int("out_channels_layer2", 4, 64, 4)
    dropout_layer1  =  trial.suggest_float("dropout_layer1", 0, 0.7)
    lin_channels = trial.suggest_int("lin_channels", 4, 64, 4)
    dropout_layer2  =  trial.suggest_float("dropout_layer2", 0, 0.7)
    model = HomoGNN(out_channels_layer1, dropout_layer1, out_channels_layer2, dropout_layer2, lin_channels).to(device)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr)

    #  Training of the model.
    for epoch in range(epochs):
        tr_loss = train(model, device, optimizer, dataloader = None, data = tr_g, loss_f = loss_f)
        val_loss = evaluate(model, device, optimizer, dataloader = None, data = val_g, loss_f = loss_f)
        trial.report(val_loss, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return val_loss