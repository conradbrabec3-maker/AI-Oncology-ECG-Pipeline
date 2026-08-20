import base64
import requests
import sys

def main():
    # 1. Define the compact, vertically-squeezed V9 Architecture
    mermaid_code = """
    %%{init: {"flowchart": {"nodespacing": 15, "rankspacing": 20}}}%%
    graph TD
        %% Global Input
        Global_Input([12-Lead ECG])

        %% Subgraph 1: Branch A
        subgraph Branch_A [Branch A: CNN Extractor]
            direction TD
            CNN["1D-CNN (BatchNorm + ReLU)"]
            Features_A["Spatial-Temporal Features"]
        end

        %% Subgraph 2: SE Attention
        subgraph SE_Attention [SE Attention]
            direction TD
            SE["SE Attention (Noise Filter)"]
            Flatten["Flatten"]
            Bottleneck["Linear Bottleneck"]
        end

        %% Subgraph 3: Branch B
        subgraph Branch_B [Branch B: HRV Timing]
            direction TD
            Lead_II["Lead II"]
            Peak_Detect["SciPy Peak Detection"]
            HRV_Metrics[/"HRV Metrics (Mean N-N, SDNN, RMSSD)"/]
        end

        %% Subgraph 4: Multimodal Fusion & Classification Head
        subgraph Fusion_Head [Fusion & Classification]
            direction TD
            Concat{"Concatenate"}
            Dense["Dense + Dropout"]
            Sigmoid("Sigmoid")
            Final_Output([Probability Output])
        end

        %% Define the flow/connections
        Global_Input --> CNN
        CNN --> Features_A
        Features_A --> SE
        SE --> Flatten
        Flatten --> Bottleneck

        Global_Input --> Lead_II
        Lead_II --> Peak_Detect
        Peak_Detect --> HRV_Metrics

        Bottleneck --> Concat
        HRV_Metrics --> Concat
        
        Concat --> Dense
        Dense --> Sigmoid
        Sigmoid --> Final_Output
        
        %% Minimalist styling for a clean, professional look
        classDef primary fill:#e1f5fe,stroke:#333,stroke-width:2px;
        classDef process fill:#fff3e0,stroke:#333,stroke-width:2px;
        classDef attention fill:#fce4ec,stroke:#333,stroke-width:2px;
        classDef output fill:#e8f5e9,stroke:#333,stroke-width:2px;
        
        class Global_Input,Lead_II primary;
        class CNN,Peak_Detect,Flatten,Bottleneck,Dense process;
        class SE,Concat attention;
        class Final_Output output;
    """

    print("⏳ Encoding tightly-packed Mermaid.js graph...")
    
    # 2. Encode the string to Base64 so the API can process it safely
    # Using urlsafe_b64encode to avoid any URL parsing issues with the API
    encoded_graph = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
    
    # 3. Construct the API request (requesting a clean white background)
    api_url = f"https://mermaid.ink/img/{encoded_graph}?bgColor=white"
    
    print("🚀 Requesting rendering from Mermaid.ink API...")
    try:
        # 15-second timeout to handle any network delays from the ARC cluster
        response = requests.get(api_url, timeout=15)
        
        if response.status_code == 200:
            output_filename = "v9_architecture_vertical_compact.png"
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ Success! Vertical flowchart securely saved to ARC storage as '{output_filename}'")
        else:
            print(f"❌ Error {response.status_code}: Could not generate image. The API may be busy or the syntax is malformed.")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: Could not reach the API from the ARC cluster. {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
