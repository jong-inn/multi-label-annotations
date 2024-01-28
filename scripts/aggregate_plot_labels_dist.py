import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from utils import DATASET_LIST, DATASET_LABELS, calculate_majority

STRING_TO_INT = {
    "'0.0'": 0,
    "'1.0'": 1,
    "'2.0'": 2,
    "'3.0'": 3,
    "'4.0'": 4,
    "'5.0'": 5
}


def main(
    dataset: str,
    base_model: str,
    plot_type: str,
    sum_constraint: bool
):
    
    test_df = load_test_df(dataset)
    test_df["idx"] = test_df.index
    
    prediction_path = Path(__file__).parent.parent.joinpath("prediction")
    prediction_human = pd.read_csv(prediction_path.joinpath(f"{dataset}_human_{base_model}_{sum_constraint}_labels_dist_prediction.csv"))
    prediction_model = pd.read_csv(prediction_path.joinpath(f"{dataset}_model_{base_model}_{sum_constraint}_labels_dist_prediction.csv"))
    
    
    prediction_human["prediction"] = prediction_human["prediction"].str.strip("[]").str.split(",").apply(lambda x: [int(i.strip()) for i in x])
    prediction_model["prediction"] = prediction_model["prediction"].str.strip("[]").str.split(",").apply(lambda x: [int(i.strip()) for i in x])
    
    test_df["human_predictions"] = prediction_human["prediction"]
    test_df["model_predictions"] = prediction_model["prediction"]
    
    rows = []
    labels_num = len(test_df.loc[0, "human_label_count"])
    
    for label in range(labels_num):
        ho_dic = {"annotation_type": "HO", "annotation": label, "count": sum(map(lambda x: x[label], test_df["human_label_count"]))}
        hp_dic = {"annotation_type": "HP", "annotation": label, "count": sum(map(lambda x: x[label], test_df["human_predictions"]))}
        mo_dic = {"annotation_type": "MO", "annotation": label, "count": sum(map(lambda x: x[label], test_df["model_label_count"]))}
        mp_dic = {"annotation_type": "MP", "annotation": label, "count": sum(map(lambda x: x[label], test_df["model_predictions"]))}
        
        rows.append(ho_dic)
        rows.append(hp_dic)
        rows.append(mo_dic)
        rows.append(mp_dic)
        
    exploded_df = pd.DataFrame(rows)
    
    human_concat = []
    model_concat = []
    for label in range(labels_num):
        test_df[f"human_head_{label}_original"] = test_df["human_label_count"].apply(lambda x: x[label])
        test_df[f"human_head_{label}_prediction"] = test_df["human_predictions"].apply(lambda x: x[label])
        test_df[f"model_head_{label}_original"] = test_df["model_label_count"].apply(lambda x: x[label])
        test_df[f"model_head_{label}_prediction"] = test_df["model_predictions"].apply(lambda x: x[label])
        
        # Human original
        human_head_original_tmp = test_df[["text_ind", f"human_head_{label}_original"]]
        human_head_original_tmp = human_head_original_tmp.groupby(f"human_head_{label}_original").count()
        human_head_original_tmp.reset_index(drop=False, inplace=True)
        human_head_original_tmp["head_num"] = f"head_{label}"
        human_head_original_tmp.rename(columns={"text_ind": "count", f"human_head_{label}_original": "human_head_label"}, inplace=True)

        no_count = [label for label in range(labels_num) if label not in human_head_original_tmp[f"human_head_label"].unique()]
        for n in no_count:
            human_head_original_tmp.loc[len(human_head_original_tmp.index)] = [n, 0, f"head_{label}"]
        human_head_original_tmp["annotation_type"] = "HO"
        
        # Human prediction
        human_head_prediction_tmp = test_df[["text_ind", f"human_head_{label}_prediction"]]
        human_head_prediction_tmp = human_head_prediction_tmp.groupby(f"human_head_{label}_prediction").count()
        human_head_prediction_tmp.reset_index(drop=False, inplace=True)
        human_head_prediction_tmp["head_num"] = f"head_{label}"
        human_head_prediction_tmp.rename(columns={"text_ind": "count", f"human_head_{label}_prediction": "human_head_label"}, inplace=True)
        
        no_count = [label for label in range(labels_num) if label not in human_head_prediction_tmp[f"human_head_label"].unique()]
        for n in no_count:
            human_head_prediction_tmp.loc[len(human_head_prediction_tmp.index)] = [n, 0, f"head_{label}"]
        human_head_prediction_tmp["annotation_type"] = "HP"
        
        # Human concatenation
        human_head_tmp = pd.concat([human_head_original_tmp, human_head_prediction_tmp], ignore_index=True)
        human_concat.append(human_head_tmp.copy())

        # Model original
        model_head_original_tmp = test_df[["text_ind", f"model_head_{label}_original"]]
        model_head_original_tmp = model_head_original_tmp.groupby(f"model_head_{label}_original").count()
        model_head_original_tmp.reset_index(drop=False, inplace=True)
        model_head_original_tmp["head_num"] = f"head_{label}"
        model_head_original_tmp.rename(columns={"text_ind": "count", f"model_head_{label}_original": "model_head_label"}, inplace=True)
        
        no_count = [label for label in range(labels_num) if label not in model_head_original_tmp[f"model_head_label"].unique()]
        for n in no_count:
            model_head_original_tmp.loc[len(model_head_original_tmp.index)] = [n, 0, f"head_{label}"]
        model_head_original_tmp["annotation_type"] = "MO"

        # Model prediction
        model_head_prediction_tmp = test_df[["text_ind", f"model_head_{label}_prediction"]]
        model_head_prediction_tmp = model_head_prediction_tmp.groupby(f"model_head_{label}_prediction").count()
        model_head_prediction_tmp.reset_index(drop=False, inplace=True)
        model_head_prediction_tmp["head_num"] = f"head_{label}"
        model_head_prediction_tmp.rename(columns={"text_ind": "count", f"model_head_{label}_prediction": "model_head_label"}, inplace=True)
        
        no_count = [label for label in range(labels_num) if label not in model_head_prediction_tmp[f"model_head_label"].unique()]
        for n in no_count:
            model_head_prediction_tmp.loc[len(model_head_prediction_tmp.index)] = [n, 0, f"head_{label}"]
        model_head_prediction_tmp["annotation_type"] = "MP"
        
        # Model concatenation
        model_head_tmp = pd.concat([model_head_original_tmp, model_head_prediction_tmp], ignore_index=True)
        model_concat.append(model_head_tmp.copy())
        
    # human_concat_df = pd.concat(human_concat, ignore_index=True)
    # human_concat_df["human_head_label"] = human_concat_df["human_head_label"].apply(str)
    # model_concat_df = pd.concat(model_concat, ignore_index=True)
    # model_concat_df["model_head_label"] = model_concat_df["model_head_label"].apply(str)
    
    
    # # Explosion
    # human_original_exploded = test_df[["text_ind", "human_label_count"]]
    # human_original_exploded.rename(columns={"human_label_count": "annotation"}, inplace=True)
    # human_original_exploded = human_original_exploded.explode("annotation")
    # human_original_exploded["annotation_type"] = "HO"
    
    # human_hypothesis_exploded = test_df[["text_ind", "human_predictions"]]
    # human_hypothesis_exploded.rename(columns={"human_predictions": "annotation"}, inplace=True)
    # human_hypothesis_exploded = human_hypothesis_exploded.explode("annotation")
    # human_hypothesis_exploded["annotation_type"] = "HP"
    
    # model_original_exploded = test_df[["text_ind", "model_label_count"]]
    # model_original_exploded.rename(columns={"model_label_count": "annotation"}, inplace=True)
    # model_original_exploded = model_original_exploded.explode("annotation")
    # model_original_exploded["annotation_type"] = "MO"
    
    # model_hypothesis_exploded = test_df[["text_ind", "model_predictions"]]
    # model_hypothesis_exploded.rename(columns={"model_predictions": "annotation"}, inplace=True)
    # model_hypothesis_exploded = model_hypothesis_exploded.explode("annotation")
    # model_hypothesis_exploded["annotation_type"] = "MP"
    
    # exploded_df = pd.concat(
    #     [human_original_exploded, human_hypothesis_exploded, model_original_exploded, model_hypothesis_exploded],
    #     ignore_index=True
    # )
    
    # test_df["human_majority"] = test_df.apply(lambda row: calculate_majority(row["idx"], row["human_annots"]), axis=1)
    # test_df["model_aggregated_majority"] = test_df.apply(lambda row: calculate_majority(row["idx"], row["model_majority"]), axis=1)
    # test_df["human_predictions_majority"] = test_df.apply(lambda row: calculate_majority(row["idx"], row["human_predictions"]), axis=1)
    # test_df["model_predictions_majority"] = test_df.apply(lambda row: calculate_majority(row["idx"], row["model_predictions"]), axis=1)
    
    # melted_test_df = test_df[["text_ind", "human_majority", "model_aggregated_majority", "human_predictions_majority", "model_predictions_majority"]]
    # melted_test_df.rename(columns={"human_majority": "HO", "model_aggregated_majority": "MO", "human_predictions_majority": "HP", "model_predictions_majority": "MP"}, inplace=True)
    # melted_test_df = melted_test_df.melt(id_vars=["text_ind"], var_name="annotation_type", value_name="annotation")
    
    # print(test_df.head())
    # print(melted_test_df.head())
    print(exploded_df.head())
    
    if plot_type == "displot":
        pass
        # fig1, ax1 = plt.subplots()
        # sns.displot(
        #     ax=ax1,
        #     data=melted_test_df, 
        #     x="annotation", 
        #     hue="annotation_type", 
        #     kind="kde", 
        #     # facet_kws={"hue_kws": {"color": ["C0", "C0", "k", "k"], "ls": ["-", "--", "-", "--"]}}
        #     palette=sns.color_palette("bright")[:4]
        # )
        # ax1.set_ylim(-0.1, 10)
        # plt.savefig(Path(__file__).parent.parent.joinpath("plot", f"{dataset}_{base_model}_{plot_type}_labels_dist_majority.pdf"), format="pdf")
        
        # fig2, ax2 = plt.subplots()
        # sns.displot(
        #     ax=ax2,
        #     data=exploded_df, 
        #     x="annotation", 
        #     hue="annotation_type", 
        #     kind="kde", 
        #     # kde_kws={"linestyle": ["-", "--", "-", "--"]}
        #     # facet_kws={"hue_kws": {"color": ["C0", "C0", "k", "k"], "ls": ["-", "--", "-", "--"]}}
        #     palette=sns.color_palette("bright")[:4]
        # )
        # ax2.set_ylim(-0.1, 10)
        # plt.savefig(Path(__file__).parent.parent.joinpath("plot", f"{dataset}_{base_model}_{plot_type}_labels_dist_exploded.pdf"), format="pdf")
    elif plot_type == "line":
        # melted_line_df = melted_test_df.groupby(["annotation_type", "annotation"]).count()
        # melted_line_df = melted_line_df.reset_index()
        # melted_line_df.rename(columns={"text_ind": "count"}, inplace=True)
        # melted_line_df["annotation"] = melted_line_df["annotation"].apply(str)
        
        # for annotation_type in ["HO", "HP", "MO", "MP"]:
        #     for label in DATASET_LABELS[dataset]:
        #         filtered = melted_line_df.loc[(melted_line_df["annotation_type"] == annotation_type) & (melted_line_df["annotation"] == str(label))]
        #         if len(filtered) == 0:
        #             # append a row
        #             melted_line_df.loc[len(melted_line_df.index)] = [annotation_type, str(label), 0]
        
        # melted_line_df.sort_values(by=["annotation_type", "annotation"], inplace=True)
        
        # print(melted_line_df)
        
        # exploded_line_df = exploded_df.groupby(["annotation_type", "annotation"]).count()
        # exploded_line_df = exploded_line_df.reset_index()
        # exploded_line_df.rename(columns={"text_ind": "count"}, inplace=True)
        # exploded_line_df["annotation"] = exploded_line_df["annotation"].apply(str)
        
        exploded_line_df = exploded_df
        exploded_line_df = exploded_line_df.reset_index()
        exploded_line_df["annotation"] = exploded_line_df["annotation"].apply(str)
        
        # for annotation_type in ["HO", "HP", "MO", "MP"]:
        #     labels = DATASET_LABELS[dataset]
        #     labels.append(5)
        #     for label in DATASET_LABELS[dataset]:
        #         filtered = exploded_line_df.loc[(exploded_line_df["annotation_type"] == annotation_type) & (exploded_line_df["annotation"] == str(label))]
        #         print(f"annotation_type: {annotation_type}, label: {label}, filtered length: {len(filtered)}")
        #         if len(filtered) == 0:
        #             # append a row
        #             exploded_line_df.loc[len(exploded_line_df.index)] = [annotation_type, str(label), 0]
        
        exploded_line_df.to_json("test.json")
           
        colors = [
            sns.color_palette(palette="light:red", desat=0.5)[2], 
            sns.color_palette(palette="light:red")[-1],
            sns.color_palette(palette="light:blue", desat=0.5)[2],
            sns.color_palette(palette="light:blue")[-1]
        ]

        # For legend
        line1 = Line2D([0,8],[0,8],linestyle="-", color=colors[0])
        line2 = Line2D([0,8],[0,8],linestyle="--", color=colors[1])
        line3 = Line2D([0,8],[0,8],linestyle="-", color=colors[2])
        line4 = Line2D([0,8],[0,8],linestyle="--", color=colors[3])
                
        # fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        # sns.lineplot(data=melted_line_df, x="annotation", y="count", hue="annotation_type", palette=colors, estimator=None, linewidth=3)
        # ax1.lines[1].set_linestyle("--")
        # ax1.lines[3].set_linestyle("--")
        
        # ax1.set_xlabel("Counts",fontsize=24)
        # ax1.set_ylabel("Labels",fontsize=24)
        # ax1.set_xticklabels(ax1.get_xticks(), size = 20)
        # ax1.set_yticklabels(ax1.get_yticks(), size = 20)
        # # plt.setp(ax1.get_legend().get_title(), fontsize="24") # for legend title
        # # plt.legend(loc='upper left')
        # ax1.legend([line1, line2, line3, line4], ["HO", "HP", "MO", "MP"], loc="upper left")
        # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
        # plt.setp(ax1.get_legend().get_texts(), fontsize="24") # for legend text
        # plt.setp(ax1.get_legend().get_lines(), linewidth=3) # for legend line
        # # ax1.set_title(f"{dataset} - majority")
        
        # plt.savefig(Path(__file__).parent.parent.joinpath("plot", f"{dataset}_{base_model}_{plot_type}_labels_dist_majority.pdf"), format="pdf")
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
        sns.lineplot(data=exploded_line_df, x="annotation", y="count", hue="annotation_type", palette=colors, estimator=None, linewidth=3)
        ax2.lines[1].set_linestyle("--")
        ax2.lines[3].set_linestyle("--")
        ax2.set_xlabel("Label",fontsize=24)
        ax2.set_ylabel("Counts",fontsize=24)
        ax2.set_xticklabels(ax2.get_xticks(), size = 20)
        ax2.set_yticklabels(ax2.get_yticks(), size = 20)
        # plt.setp(ax2.get_legend().get_title(), fontsize="24") # for legend title
        # plt.legend(loc='upper left')
        ax2.legend([line1, line2, line3, line4], ["HO", "HP", "MO", "MP"], loc="upper left")
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
        plt.setp(ax2.get_legend().get_texts(), fontsize="24") # for legend text
        plt.setp(ax2.get_legend().get_lines(), linewidth=3) # for legend line
        # ax2.set_title(f"{dataset} - exploded")
        
        plt.savefig(Path(__file__).parent.parent.joinpath("plot", f"{dataset}_{base_model}_{plot_type}_{sum_constraint}_labels_dist_exploded.pdf"), format="pdf")
        
    elif plot_type == "bar":
        colors = [
            sns.color_palette(palette="light:red", desat=0.5)[2], 
            sns.color_palette(palette="light:red")[-1],
            sns.color_palette(palette="light:blue", desat=0.5)[2],
            sns.color_palette(palette="light:blue")[-1]
        ]
        
        # Human
        fig1, ax1 = plt.subplots(1, 5, figsize=(15, 10), sharey=True)
        for idx, head_df in enumerate(human_concat):
            sns.barplot(ax=ax1[idx], data=head_df, x="human_head_label", y="count", hue="annotation_type", palette=colors[:2])
            if idx != 4:
                ax1[idx].get_legend().remove()
            else:
                ax1[idx].get_legend().set_title(None)
                plt.setp(ax1[idx].get_legend().get_texts(), fontsize="24")
        
            if idx == 0:
                ax1[idx].set_ylabel("Counts",fontsize=24)
                ax1[idx].set_yticklabels(ax1[idx].get_yticks(), size = 14)
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
            else:
                ax1[idx].set_ylabel(None)
            ax1[idx].set_xlabel(f"head_{idx}",fontsize=20)
            ax1[idx].set_xticklabels(ax1[idx].get_xticks(), size = 12)
        
        plt.savefig(Path(__file__).parent.parent.joinpath("plot", f"{dataset}_{base_model}_{plot_type}_{sum_constraint}_human_head_labels_dist.pdf"), format="pdf")
        
        # Model
        fig2, ax2 = plt.subplots(1, 5, figsize=(15, 10), sharey=True)
        for idx, head_df in enumerate(model_concat):
            sns.barplot(ax=ax2[idx], data=head_df, x="model_head_label", y="count", hue="annotation_type", palette=colors[2:])
            if idx != 4:
                ax2[idx].get_legend().remove()
            else:
                ax2[idx].get_legend().set_title(None)
                plt.setp(ax2[idx].get_legend().get_texts(), fontsize="24")
            
            if idx == 0:
                ax2[idx].set_ylabel("Counts",fontsize=24)
                ax2[idx].set_yticklabels(ax2[idx].get_yticks(), size = 14)
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
            else:
                ax2[idx].set_ylabel(None)
            ax2[idx].set_xlabel(f"head_{idx}",fontsize=20)
            ax2[idx].set_xticklabels(ax2[idx].get_xticks(), size = 12)
            
        plt.savefig(Path(__file__).parent.parent.joinpath("plot", f"{dataset}_{base_model}_{plot_type}_{sum_constraint}_model_head_labels_dist.pdf"), format="pdf")
        
        
    elif plot_type == "hist":
        pass
    

def load_test_df(dataset: str) -> pd.DataFrame:
    data_path = Path(__file__).parent.parent.joinpath("data")
    
    test_df = pd.read_csv(data_path.joinpath(f"df_final_{dataset}_test.csv"))
    
    test_df["human_annots"] = test_df["human_annots"].str.strip("[]").str.split(",").apply(lambda x: [STRING_TO_INT[i.strip()] for i in x])
    test_df["model_majority"] = test_df["model_majority"].str.strip("[]").str.split(",").apply(lambda x: [STRING_TO_INT[i.strip()] for i in x])
    
    def annotation_to_count(annotations):
        count = {label: 0 for label in DATASET_LABELS[dataset]}
        for annotation in annotations:
            count[annotation] += 1
        return list(count.values())
    
    test_df["human_label_count"] = test_df["human_annots"].apply(lambda x: annotation_to_count(x))
    test_df["model_label_count"] = test_df["model_majority"].apply(lambda x: annotation_to_count(x))
    
    print(f"""
dataset: {dataset}
shape: {test_df.shape}
head:
    {test_df.head()}
          """)
    
    return test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="SChem5Labels",
        choices=DATASET_LIST
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="roberta-large"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="line"
    )
    parser.add_argument(
        "--sum_constraint",
        action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()
    if not args.sum_constraint:
        sum_constraint = False
    else:
        sum_constraint = True
    
    args = parser.parse_args()
    
    main(
        args.dataset,
        args.base_model,
        args.plot_type,
        sum_constraint
    )