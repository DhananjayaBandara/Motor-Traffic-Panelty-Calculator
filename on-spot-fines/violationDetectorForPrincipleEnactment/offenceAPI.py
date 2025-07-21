from flask import Flask, request, Response
import pandas as pd

app = Flask(__name__)

# Load the fines data
df = pd.read_csv('IndexFines.csv')

@app.route('/get_fines', methods=['POST'])
def get_fines():
    try:
        # Get input data
        data = request.get_json()
        indexes_input = data.get('indexes', '').strip()
        
        # Handle 'None' case
        if indexes_input.lower() == 'none':
            indexes = []
        else:
            # Split and clean indexes
            indexes = [idx.strip() for idx in indexes_input.split(',') if idx.strip()]
        
        # Filter dataframe
        result_df = df[df['Index'].isin(indexes)]
        
        # Create CSV output
        if result_df.empty:
            csv_output = "Index,Offence,Fine\n"
        else:
            csv_output = result_df.to_csv(index=False)
        
        return Response(
            csv_output,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment;filename=fines_result.csv'}
        )
    
    except Exception as e:
        return Response(
            f"Error: {str(e)}",
            mimetype='text/plain',
            status=500
        )

if __name__ == '__main__':
    app.run(debug=True)


"""

http://localhost:5000/get_fines

{
    "indexes": "o3"
}

{
    "indexes": "o21, o14"
}

{
    "indexes": "None"
}

"""