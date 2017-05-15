# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import NHS inpatient and outpatient data, skipping irrelevant columns
df = pd.read_csv('./data/QAR-COMM-Timeseries-1617-Q3-50207.csv', index_col=[0,1], skiprows=13, usecols=[x for x in range(1,17) if x not in [3,4,9,10]])
print(df.index)
print(df)

# Plot df for first visualisation
df.plot()
plt.title('NHS Overview')
plt.ylabel('Patients in Million per Quarter')
plt.xticks(rotation=45)
plt.savefig('./plots/all_data_raw.png')
plt.show(block=True)
plt.close()

# Run correlation analysis based on pearson standard correlation coefficient
pearson = df.corr(method='pearson')
print(pearson)

# Visualise correlation
hm = sns.heatmap(pearson)
plt.title('Correlation Heatmap (1 = high positive correlation)')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig('./plots/correlation_heatmap.png')
plt.show(block=True)
plt.close()

# Check change rates
pct = df.pct_change()
print(pct)

# Get average change rate from december to march
df_rem_last = df
df_rem_last = df_rem_last.drop(('2016-17', 'DECEMBER'))
print(df_rem_last)

sum_by_q = df_rem_last.groupby(level=1, sort=False).sum()
sum_by_q = sum_by_q.reindex(['DECEMBER', 'MARCH', 'JUNE', 'SEPTEMBER'])

pct_av = sum_by_q.pct_change()
print(pct_av)

mar_pct_av = pct_av.query('Period == "MARCH"')
print(mar_pct_av)

# =================
# Check yearly data

# Get sums Q1:Q4
yd = df.groupby(level=0).sum()
print(yd)

# Visualize sums Q1:Q4 to get first impression without seasonal fluctuations
yd['Admissions'].plot()
plt.title('NHS Inpatient Admissions')
plt.ylabel('Patients per Financial Year')
ax = plt.gca()
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig('./plots/inpatient_admissions_yearly_sums.png')
plt.show(block=True)
plt.close()

# First very rudimentary forecast for March 2017
dec16 = df.loc['2016-17'].loc['DECEMBER']
print(dec16)
mar17_pred = dec16 * (1 + mar_pct_av)
mar17_pred['Year'] = '2016-17'
mar17_pred.set_index('Year', append=True, inplace=True)
mar17_pred = mar17_pred.reorder_levels(['Year', 'Period'])
print('mar17_pred: ', mar17_pred)
print(mar17_pred.index)

# Then add it to the complete dataframe
df_pred = df
df_pred = df_pred.append(mar17_pred)
print(df_pred)


# Full years incl Mar 2017 prediction
yd_pred = df_pred.groupby(level=0).sum()
print(yd_pred)


