# importing libraries
import pandas as pd
import numpy as np
import datetime
from datetime import date, timedelta
import warnings
import streamlit as st
import xlsxwriter
import io
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Transactions", page_icon="🛒")
st.title("Transaction Breakdown")

# streamlit app building
filename = st.text_input("Filename", key="filename")
firstname = st.text_input("Enter Name", key="firstname1")
highticket = st.number_input("Enter high ticket (integer only)", key="highticket")
uploaded_file = st.file_uploader("Please upload a CSV file", type=['csv'])

if uploaded_file is not None:
    highticketval = int(highticket)
    dfpreclean = pd.read_csv(uploaded_file)

    buffer = io.BytesIO()

    dfpreclean.drop(['Transaction_ID', 'Auth_code'], axis=1, inplace=True)
    dfpreclean2 = dfpreclean[dfpreclean['Success'] == 1]
    dfpreclean2['Transaction_Notes'].fillna("N/A", inplace=True)
    dfpreclean2['Day'] = pd.to_datetime(dfpreclean2['Day'])  

    df = dfpreclean2.loc[:, ['Total', 'Transaction_Type', 'Type', 'Country', 'Source', 
                            'Day', 'Customer_Name', 'Transaction_Notes' ]]

    # Calculations

    totalsum = np.sum(df['Total'])
    total_transactions = df['Type'].count()

    chargeonlytransactions = df[df['Type'] == 'Charge']
    refundonlytransactions = df[df['Type'] == 'Refund']
    chargebackonlytransactions = df[df['Type'] == 'Chargeback']

    mean_transaction = np.mean(df['Total'])
    median_transaction = np.median(df['Total'])
    max_transaction = np.max(df['Total'])

    total_unique_customers = df['Customer_Name'].nunique()
    days90 = pd.to_datetime(date.today() - timedelta(days=90))
    days180 = pd.to_datetime(date.today() - timedelta(days=180))

    # charge
    chargetotal = np.sum(chargeonlytransactions['Total'])
    charge90days = np.sum(chargeonlytransactions[chargeonlytransactions['Day']> days90]['Total'])
    charge180days = np.sum(chargeonlytransactions[chargeonlytransactions['Day']> days180]['Total'])

    # refund
    refundtotal = np.sum(refundonlytransactions['Total'])
    refund90days = np.sum(refundonlytransactions[refundonlytransactions['Day']> days90]['Total'])
    refund180days = np.sum(refundonlytransactions[refundonlytransactions['Day']> days180]['Total'])

    # chargeback
    chargebacktotal = np.sum(chargebackonlytransactions['Total'])
    chargeback90days = np.sum(chargebackonlytransactions[chargebackonlytransactions['Day']> days90]['Total'])
    chargeback180days = np.sum(chargebackonlytransactions[chargebackonlytransactions['Day']> days180]['Total'])

    # refund rates
    refundratelifetime = (refundtotal/chargetotal)
    refundrate90days = (refund90days/charge90days)
    refundrate180days = (refund180days/charge180days)

    # chargeback rates
    chargebackratelifetime = (chargebacktotal/chargetotal)
    chargebackrate90days = (chargeback90days/charge90days)
    chargebackrate180days = (chargeback180days/charge180days)


    # pivot tables
    pivottablenames = pd.pivot_table(df, index=['Customer_Name'], 
                                    aggfunc={'Total': np.sum, 'Customer_Name': 'count'})
    pivottablenames = pivottablenames.rename(columns={'Customer_Name': "count_of_total",
                                                    'Total':'sum_of_total'})
    pivottablenames = pivottablenames.loc[:, ['sum_of_total', 'count_of_total']]

    avg_transactions_count_per_customer = np.mean(pivottablenames['count_of_total'])
    avg_transactions_sum_per_customer = np.mean(pivottablenames['sum_of_total'])

    pivottabletransactiontype = pd.pivot_table(df, index=['Transaction_Type'], 
                                    aggfunc={'Total': np.sum, 'Transaction_Type': 'count'})
    pivottabletransactiontype['totalpercent'] = (pivottabletransactiontype['Total']/totalsum).apply('{:.2%}'.format)

    pivottabletransactioncountry = pd.pivot_table(df, index=['Country'], 
                                    aggfunc={'Country': 'count', 'Total': np.sum})
    pivottabletransactioncountry['totalpercent'] = (pivottabletransactioncountry['Total']/totalsum).apply('{:.2%}'.format)

    namefinal = df[df['Customer_Name'].str.contains(firstname, case=False)]

    # Notes
    payment_note = df[df['Transaction_Notes'].isna()==False]
    flagged_words = 'raffle|razz|lottery'
    payment_note_final = df[df['Transaction_Notes'].str.contains(flagged_words, case=False)]

    highticket = df[df['Total'] >= highticketval].copy()
    highticket = highticket.sort_values(by='Total', ascending=False)

    #
    dup = df.copy()

    dup['Customer_Name_next'] = dup['Customer_Name'].shift(1)
    dup['Customer_Name_prev'] = dup['Customer_Name'].shift(-1)

    dup['created_at_day'] = dup['Day']
    dup['created_at_dayprev'] = dup['Day'].shift(-1)
    dup['created_at_daynext'] = dup['Day'].shift(1)

    dup2 = dup.query('(Customer_Name == Customer_Name_next | Customer_Name == Customer_Name_prev) & (created_at_day == created_at_dayprev | created_at_day == created_at_daynext)')

    # df with all the calculations
    dfcalc = pd.DataFrame({
            'totalsum': [totalsum],
            'mean_transaction': [mean_transaction],
            'median_transaction': [median_transaction],
            'max_transaction': [max_transaction],
            'total_transactions': [total_transactions],
            'chargetotal': [chargetotal],
            'charge90days': [charge90days],
            'charge180days': [charge180days],
            'refundtotal': [refundtotal],
            'refund90days': [refund90days],
            'refund180days': [refund180days],
            'chargebacktotal': [chargebacktotal],
            'chargeback90days': [chargeback90days],
            'chargeback180days': [chargeback180days],
            'refundratelifetime': [refundratelifetime],
            'refundrate90days': [refundrate90days],
            'refundrate180days': [refundrate180days],
            'chargebackratelifetime': [chargebackratelifetime],
            'chargebackrate90days': [chargebackrate90days],
            'chargebackrate180days': [chargebackrate180days],
            'total_unique_customers': [total_unique_customers],
            'avg_transaction_count_per_customer_name': [avg_transactions_count_per_customer],
            'avg_transaction_sum_per_customer_name': [avg_transactions_sum_per_customer],
            '90_Days': [days90],
            '180_Days': [days180]
        })

    format_mapping = {
            'totalsum': '${:,.2f}',
            'mean_transaction': '${:,.2f}',
            'median_transaction': '${:,.2f}',
            'max_transaction': '${:,.2f}',
            'total_transactions': '{:,.0f}',
            'chargetotal': '${:,.2f}',
            'charge90days': '${:,.2f}',
            'charge180days': '${:,.2f}',
            'refundtotal': '${:,.2f}',
            'refund90days': '${:,.2f}',
            'refund180days': '${:,.2f}',
            'chargebacktotal': '${:,.2f}',
            'chargeback90days': '${:,.2f}',
            'chargeback180days': '${:,.2f}',
            'refundratelifetime': '{:.2%}',
            'refundrate90days': '{:.2%}',
            'refundrate180days': '{:.2%}',
            'chargebackratelifetime': '{:.2%}',
            'chargebackrate90days': '{:.2%}',
            'chargebackrate180days': '{:.2%}',
            'total_unique_customers': '{:,.0f}',
            'avg_transaction_count_per_customer_name': '{:,.2f}',
            'avg_transaction_sum_per_customer_name': '${:,.2f}' ,
        }

    for key, value in format_mapping.items():
        dfcalc[key] = dfcalc[key].apply(value.format)
        

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Clean_Data')
        dfcalc.to_excel(writer, sheet_name='Calculations')
        pivottablenames.to_excel(writer, sheet_name='Names')
        pivottabletransactiontype.to_excel(writer, sheet_name='Transaction_Type')
        pivottabletransactioncountry.to_excel(writer, sheet_name='Countries')
        payment_note_final.to_excel(writer, sheet_name='Payment_Notes')
        highticket.to_excel(writer, sheet_name='High_Ticket')
        namefinal.to_excel(writer, sheet_name='Name_Checker')
        dup2.to_excel(writer, sheet_name='Double_Transactions')
        
        writer.close()

    st.download_button(
        label = 'Download Excel File',
        data=buffer,
        file_name= f"{st.session_state.filename}.xlsx",
        mime = "application/vnd.ms-excel"
    )

else:
    st.warning("You need to upload a csv")