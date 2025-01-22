import solara
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from bfio import BioReader
from datetime import datetime
import pyarrow.ipc as ipc
import pyarrow as pa
import os
import matplotlib
import io
from PIL import Image



def load_data() -> pd.DataFrame:
    """
    Load the dataset from an Arrow file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    path = '/opt/shared/notebooks/2025_PanMicroscopy_dashboard/dashboard/T-SNE_PanMicroscopy_modified.arrow'
    with pa.memory_map(path, 'r') as source:
        table = ipc.open_file(source).read_all()

    df = table.to_pandas()

    return df

# Load the data

df = load_data()

# Reactive variables
selected_dataset = solara.reactive("All")
selected_points = solara.reactive([])
filtered_df = solara.reactive(df)
show_table = solara.reactive(False)  
show_images = solara.reactive(False)  
filtered_df_with_indices = solara.reactive(None)


def save_dataframe(df: pd.DataFrame, filename: str):
    """
    Save the DataFrame as a CSV file.

    Args:
        df: Pandas DataFrame to save.
        filename: Name of the file to save the DataFrame as.
    """
    df.to_csv(filename, index=False)

def save_images(selected_data: pd.DataFrame, folder: str):
    """
    Save images from selected data into a folder.

    Args:
        selected_data: DataFrame containing paths to images.
        folder: Directory to save images in.
    """
    imagelist = selected_data["path"].tolist()
    os.makedirs(folder, exist_ok=True)
    
    for i, (path, row) in enumerate(zip(imagelist, selected_data.itertuples())):
        try:
            br = BioReader(path)
            image = br.read().squeeze()
            img = Image.fromarray(image)
            img.save(os.path.join(folder, f"image_{i}_{row.dataset}_{row.cell}.png"))
        except Exception as e:
            print(f"Error saving image from {path}: {str(e)}")

def calculate_grid(n:int) -> tuple[int, ...]:
    """
    Calculate the number of rows and columns for a grid given 'n' images.

    Args:
        n: The number of images to be displayed.

    Returns:
        tuple: A tuple (rows, cols) where rows and cols are integers representing 
               the grid dimensions.
    """
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols

def plot_images(selected_data:pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Plot images from the selected data in a grid layout.

    Args:
        selected_data: DataFrame containing paths to images and metadata.

    Returns:
        A matplotlib Figure object containing the plotted images.
    """
    imagelist = selected_data["path"].tolist()
    if not imagelist:
        return plt.figure(figsize=(1, 1)) 
    
    n_rows, n_cols = calculate_grid(len(imagelist))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    ax = ax.ravel() if n_rows * n_cols > 1 else [ax]
    
    for i, (image_path, row) in enumerate(zip(imagelist, selected_data.itertuples())):
        if i >= len(ax):  # Check if there are more images than axes
            break
        try:
            br = BioReader(image_path)
            image = br.read().squeeze()
            ax[i].imshow(image, cmap="gray")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f"{row.dataset}({row.cell}; {row.channelname})")
        except Exception as e:
            print(f"Error displaying image at index {i}: {str(e)}")
            ax[i].text(0.5, 0.5, "Image not available", ha='center', va='center')
    
    for j in range(len(imagelist), len(ax)):
        fig.delaxes(ax[j])
    
    plt.tight_layout()
    return fig


def create_scatter_plot():
    """
    Create an interactive scatter plot for dataset visualization.

    Returns:
        go.FigureWidget: A Plotly FigureWidget object representing the scatter plot.
    """
    datasets = list(df["dataset"].unique())
    cmap = plt.cm.get_cmap("Spectral", len(datasets))
    color_map = {dataset: cmap(i) for i, dataset in enumerate(datasets)}
    colors = filtered_df.value["dataset"].map(color_map)

    scatter = go.Scattergl(
        x=filtered_df.value[df.columns[-2]],
        y=filtered_df.value[df.columns[-1]],
        mode="markers",
        marker=dict(opacity=0.9, size=3, color=colors),
        text=filtered_df.value[["dataset", "cell", "channelname"]].apply(
            lambda row: f"Dataset: {row['dataset']}<br>Cell: {row['cell']}<br>Channel: {row['channelname']}", axis=1
        ),
        hoverinfo="text",
        selected=dict(marker=dict(color="black")),
    )
    fig = go.FigureWidget([scatter])
    fig.update_layout(
        width=1200,
        height=600,
        xaxis=dict(title=df.columns[-2]),
        yaxis=dict(title=df.columns[-1])
    )
    return fig


def update_filtered_df(new_value:str):
    """
    Update the filtered DataFrame based on the selected dataset.

    Args:
        new_value: The name of the dataset to filter by, or "All" for all datasets.

    """

    if new_value == "All":
        filtered_df.set(df)
        filtered_df_with_indices.set(df.index.tolist())
    else:
        filtered_data = df[df["dataset"] == new_value]
        filtered_df.set(filtered_data)
        filtered_df_with_indices.set(filtered_data.index.tolist())



def clear_selection():
    """
    Clear the current selection of points, hide table and images, and reset the plot.

    Returns:
        go.FigureWidget: Updated plot with no selection.
    """
    selected_points.set([]) 
    show_table.set(False)
    show_images.set(False)
    
    fig = create_scatter_plot()
    fig.data[0].update(selectedpoints=[])
    return fig


def on_show_images_click():
    """Show the images when the button is clicked."""
    if selected_points.value:
        show_images.set(True)    
    else:
        solara.Markdown(f"**Error:** An error occurred while displaying the images.")

def on_show_table_click():
    """Show the table when the button is clicked."""
    if selected_points.value:
        show_table.set(True) 
    else:
        solara.Markdown("No points selected.")  

def on_save_click():
    """
    Handle the save button click event.

    Saves the selected data as CSV and the corresponding images to a folder.
    """
    if selected_points.value:
        selected_data = df.loc[selected_points.value]
        save_dataframe(selected_data, 'selected_data.csv')
        save_images(selected_data, 'saved_images')
        solara.Markdown("**Success:** Data and images saved successfully!")
    else:
        solara.Markdown("**Error:** No points selected to save.")

@solara.component
def Page():
    """
    Main component for the PanMicroscopy Dashboard interface.

    This component handles the UI layout, dataset selection, data visualization,
    and interaction functionalities.
    """
    def on_dataset_change(new_value:str):
        """
        Handle dataset selection change.

        Args:
            new_value: The newly selected dataset or "All" for all datasets.
        """
        selected_dataset.set(new_value)
        update_filtered_df(new_value) 

    solara.Markdown("## PanMicroscopy Dataset: A Data-Driven Approach to Unbiased Image Selection",
    style={"padding-left": "25px"}
)
    

    solara.Select(
        label="Select Dataset", 
        value=selected_dataset.value, 
        values=["All"] + list(df["dataset"].unique()), 
        on_value=on_dataset_change,
        style={"width": "200px", "padding-left": "25px"} 
    )

    def on_selection(data:dict):
        """
        Handle point selection events from the scatter plot.

        Args:
            data: Data from Plotly's selection event, containing information about selected points.

        Effects:
            - Updates 'selected_points' with the indices of selected data points.
        """
        points = data.get('points', [])
        if points: 
            trace_indexes = points.get('trace_indexes', [])
            point_indexes = points.get('point_indexes', [])
            
            if point_indexes is not None and filtered_df_with_indices.value:
                original_indices = [filtered_df_with_indices.value[i] for i in point_indexes if i < len(filtered_df_with_indices.value)]
                selected_points.set(original_indices)
            else:
                selected_points.set([])
        else:
            print("No points were selected.")
            selected_points.set([])

    fig = create_scatter_plot()
    

    solara.FigurePlotly(fig, on_selection=on_selection)
    with solara.Div(style={
        "position": "absolute",
        "right": "50px",
        "top": "30px",
        "display": "flex",
        "flex-direction": "column",
        "align-items": "flex-end",
        "z-index": "1000"
    }):
        solara.Button(
            "Clear Selection", 
            on_click=clear_selection, 
            style={
                "font-size": "10px", 
                "padding": "5px 10px", 
                "width": "150px", 
                "background-color": "#03adfc", 
                "color": "black", 
                "border": "none",
                "border-radius": "5px",
                "margin-bottom": "10px"
            }
        )

        solara.Button(
            "Show Selected Data", 
            on_click=on_show_table_click, 
            style={
                "font-size": "10px", 
                "padding": "5px 10px", 
                "width": "150px", 
                "background-color": "#98938C",
                "color": "black", 
                "border": "none",
                "border-radius": "5px",
                "margin-bottom": "10px"
            }
        )

        solara.Button(
            "Show Selected Images", 
            on_click=on_show_images_click, 
            style={
                "font-size": "10px", 
                "padding": "5px 10px", 
                "width": "150px", 
                "background-color": "#eab676",
                "color": "black", 
                "border": "none",
                "border-radius": "5px"
            }
        )
        solara.Button(
            "Save data/Images", 
            on_click=on_save_click, 
            style={
                "font-size": "10px", 
                "padding": "5px 10px", 
                "width": "150px", 
                "background-color": "#4CAF50", 
                "color": "black", 
                "border": "none",
                "border-radius": "5px",
                "margin-top": "10px"
            }
        )


    if show_table.value:
        try:
            if not selected_points.value:
                solara.Markdown("No points selected.")
            else:
                selected_data = df.loc[selected_points.value]
                solara.Markdown("## Selected Data")
                solara.DataFrame(selected_data, items_per_page=10)
        except Exception as e:
            solara.Markdown(f"**Error:** An error occurred while displaying the table: {str(e)}")

    if show_images.value:
        if selected_points.value:
            selected_data = df.loc[selected_points.value]
            fig = plot_images(selected_data)
            solara.FigureMatplotlib(fig)
        else:
            solara.Markdown("No points selected.")