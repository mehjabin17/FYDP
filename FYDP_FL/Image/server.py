import flwr as fl


def agg_metrics(metrics):
    # Collect all the FL Client metrics and weight them
    accuracy = [n_examples * metric['accuracy'] for n_examples, metric in metrics]
    loss = [n_examples * metric['loss'] for n_examples, metric in metrics]

    total_examples = sum([n_examples for n_examples, _ in metrics])

    # Compute weighted averages
    agg_metrics = {
        'accuracy': sum(accuracy) / total_examples
        , 'loss': sum(loss) / total_examples
    }

    return agg_metrics


fl_strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=agg_metrics)
result = fl.server.start_server(server_address="localhost:8080",  grpc_max_message_length=1536870912, strategy=fl_strategy, config=fl.server.ServerConfig(num_rounds=1))
print(result.metrics_distributed)