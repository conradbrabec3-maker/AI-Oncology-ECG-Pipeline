import base64
import requests
import sys

def main():
    # 1. Define the compact V9 Architecture using Mermaid.js syntax
    mermaid_code = """
    graph LR
        %% Global Input
        Global_Input([12-Lead ECG])

        %% Subgraph 1: Branch A
        subgraph Branch_A [Branch A: CNN Extractor]
            direction LR
            CNN["1D-CNN (BatchNorm + ReLU)"]
            Features_A["Spatial-Temporal Features"]
        end

        %% Subgraph 2: SE Attention
        subgraph SE_Attention [SE Attention]
            direction LR
            SE["SE Attention (Noise Filter)"]
            Flatten["Flatten"]
            Bottleneck["Linear Bottleneck"]
        end

        %% Subgraph 3: Branch B
        subgraph Branch_B [Branch B: HRV Timing]
            direction LR
            Lead_II["Lead II"]
            Peak_Detect["SciPy Peak Detection"]
            HRV_Metrics[/"HRV Metrics<br>(Mean N-N, SDNN, RMSSD)"/]
        end

        %% Subgraph 4: Multimodal Fusion & Classification Head
        subgraph Fusion_Head [Fusion & Classification]
            direction LR
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
        
        %% Minimalist styling for professional look
        classDef primary fill:#e1f5fe,stroke:#333,stroke-width:2px;
        classDef process fill:#fff3e0,stroke:#333,stroke-width:2px;
        classDef attention fill:#fce4ec,stroke:#333,stroke-width:2px;
        classDef output fill:#e8f5e9,stroke:#333,stroke-width:2px;
        
        class Global_Input,Lead_II primary;
        class CNN,Peak_Detect,Flatten,Bottleneck,Dense process;
        class SE,Concat attention;
        class Final_Output output;
    """

    print("⏳ Encoding compact Mermaid.js graph...")
    # 2. Encode the string to Base64 so the API can read it
    encoded_graph = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
    
    # 3. Construct the API request (requesting a white background)
    api_url = f"https://mermaid.ink/img/{encoded_graph}?bgColor=white"
    
    print("🚀 Requesting image render from Mermaid.ink API...")
    try:
        # 15-second timeout to be safe on the cluster's network
        response = requests.get(api_url, timeout=15)
        
        if response.status_code == 200:
            output_filename = "v9_architecture_compact.png"
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ Success! Compact flowchart securely saved to ARC storage as '{output_filename}'")
        else:
            print(f"❌ Error {response.status_code}: Could not generate image.")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: Could not reach the API from the ARC cluster. {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
