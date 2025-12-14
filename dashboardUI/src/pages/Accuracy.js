import AppMetricsPage from "../sections/@dashboard/app/AppMetricsPage";

export default function Accuracy() {
  return (
    <div>
      <div>
        <AppMetricsPage metricType="aime_2024_accuracy" title="AIME-2024 Accuracy Metrics" metricName="AIME-2024 Accuracy" showInputType={false}/>
      </div>
    </div>
  );
}
