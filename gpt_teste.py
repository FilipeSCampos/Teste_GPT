import torch
import torch.nn as nn
import torch.optim as optim
import os

# Hiperparâmetros
vocab_size = 10000  # Tamanho do vocabulário
d_model = 512       # Tamanho do vetor de embedding
nhead = 8           # Número de cabeças de atenção
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1


# Verificar se o modelo já foi treinado
if not os.path.exists('model.pt'):
    # Definir o modelo Transformer
    class TransformerModel(nn.Module):
        def __init__(self):
            super(TransformerModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.transformer = nn.Transformer(
                d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout
            )
            self.fc = nn.Linear(d_model, vocab_size)

        def forward(self, src, tgt):
            src = self.embedding(src)
            tgt = self.embedding(tgt)
            output = self.transformer(src, tgt)
            output = self.fc(output)
            return output

    # Exemplo de dados de entrada e saída
    src = torch.randint(0, vocab_size, (10, 32))  # Sequência de origem (batch_size=32, comprimento=10)
    tgt = torch.randint(0, vocab_size, (20, 32))  # Sequência de destino (batch_size=32, comprimento=20)

    # Inicializar o modelo
    model = TransformerModel()

    # Função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Loop de treinamento
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(src, tgt[:-1, :])  # O alvo deslocado é usado como entrada
        loss = criterion(output.view(-1, vocab_size), tgt[1:, :].view(-1))  # Computar a perda
        loss.backward()
        optimizer.step()
        print(f'Época {epoch + 1}, Perda: {loss.item()}')

    # Após o treinamento, salve o modelo treinado
    torch.save(model.state_dict(), 'model.pt')
else:
    print("O modelo já foi treinado e está salvo em 'model.pt'")
