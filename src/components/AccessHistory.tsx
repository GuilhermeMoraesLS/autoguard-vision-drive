import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Card } from "@/components/ui/card";
import { CheckCircle2, XCircle } from "lucide-react";

export interface AccessRecord {
  id: string;
  driver: string;
  status: "authorized" | "unauthorized";
  timestamp: string;
}

interface AccessHistoryProps {
  records: AccessRecord[];
}

export const AccessHistory = ({ records }: AccessHistoryProps) => {
  return (
    <Card className="p-6 bg-card border-border">
      <h2 className="text-xl font-bold text-foreground mb-4">Histórico de Acessos</h2>
      
      {records.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          <p>Nenhum registro ainda</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-border hover:bg-secondary/50">
                <TableHead className="text-foreground font-semibold">Motorista</TableHead>
                <TableHead className="text-foreground font-semibold">Status</TableHead>
                <TableHead className="text-foreground font-semibold">Data/Hora</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {records.map((record) => (
                <TableRow key={record.id} className="border-border hover:bg-secondary/50">
                  <TableCell className="font-medium text-foreground">
                    {record.driver}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      {record.status === "authorized" ? (
                        <>
                          <CheckCircle2 className="w-4 h-4 text-success" />
                          <span className="text-success font-medium">Autorizado</span>
                        </>
                      ) : (
                        <>
                          <XCircle className="w-4 h-4 text-danger" />
                          <span className="text-danger font-medium">Desconhecido</span>
                        </>
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {record.timestamp}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </Card>
  );
};
