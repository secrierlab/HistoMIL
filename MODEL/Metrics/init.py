import torchmetrics

metrics_dict = {
    "accuracy": torchmetrics.Accuracy,
    "cohen_kappa": torchmetrics.CohenKappa,
    "f1_score": torchmetrics.F1Score,
    "recall": torchmetrics.Recall,
    "precision": torchmetrics.Precision,
    "specificity": torchmetrics.Specificity, # class nb >2

    "auroc": torchmetrics.AUROC,
    "roc": torchmetrics.ROC,
}
metrics_para_dict = {
"classification":{
    "accuracy": {"average": "micro"},
    "cohen_kappa": {},
    "f1_score": {"average": "macro"},
    "recall": {"average": "macro"},
    "precision": {"average": "macro"},
    "specificity": {"average": "macro"},

    "auroc": {"average": "macro"},
    "roc": {"average": "macro"},
},
}


class MetricsFactory:
    def __init__(self, n_classes, metrics_names=["auroc","accuracy"]):
        """
        MetricsFactory to create metrics for training, validation and testing
        Args:
            n_classes (int): number of classes
            metrics_names (list, optional): list of metrics names. Defaults to ["auroc","accuracy"].
        
        """
        self.n_classes = n_classes
        self.metrics_names = metrics_names
        self.metrics = {}

    def get_metrics_classification(self):
        """
        get metrics for classification task
        """
        metrics_fn_list = []
        bar_metrics = self.metrics_names[0]
        assert bar_metrics in metrics_dict.keys(), f"{bar_metrics} not in metrics_dict"
        related_paras = metrics_para_dict["classification"][bar_metrics]
        self.metrics["metrics_on_bar"] = metrics_dict[bar_metrics](num_classes =self.n_classes,
                                                                    **related_paras)
        for i in range(1,len(self.metrics_names)):
            name = self.metrics_names[i]
            related_paras = metrics_para_dict["classification"][name]
            metrics_fn_list.append(
                                    metrics_dict[name](num_classes =self.n_classes,
                                                        **related_paras)
                                    )
        metrics_template = torchmetrics.MetricCollection(metrics_fn_list)
        self.metrics["metrics_template"] = metrics_template

    def get_metrics(self,task_type:str):
        if task_type == "classification":
            self.get_metrics_classification()
        else:
            raise NotImplementedError
            

