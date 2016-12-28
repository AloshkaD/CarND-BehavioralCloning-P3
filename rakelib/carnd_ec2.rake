CARND_IP = '54.245.46.71'

namespace :carnd do
  task :ssh do
    sh "ssh carnd@#{CARND_IP}"
  end

  task :pull do
    sh "ssh -t carnd@#{CARND_IP} 'cd ~/CarND-BehavioralCloning-P3 && git pull'"
  end

  task :drive do |t, args|
    sh 'cd ~/CarND-BehavioralCloning-P3 && python3 drive.py model.json'
  end

  task :scp, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~/CarND-BehavioralCloning-P3')
    host = "carnd@#{CARND_IP}"
    puts "downloading #{args[:src]} to #{host}:#{args[:dest]}"
    sh "scp -rp #{args[:src]} #{host}:#{args[:dest]}"
  end

  task :sync, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~/CarND-BehavioralCloning-P3')
    host = "carnd@#{CARND_IP}"
    puts "uploading #{args[:src]} to #{host}:#{args[:dest]}"

    sh "scp -rp driving_log.csv #{host}:~/CarND-BehavioralCloning-P3"
    sh "rsync -ravz --progress --ignore-existing IMG #{host}:~/CarND-BehavioralCloning-P3"

    unless args[:src].nil?
      sh "rsync -avvz --update --existing --ignore-existing #{args[:src]} #{host}:#{args[:dest]}"
    end
  end

  task :get_model, [] do
    host = "carnd@#{CARND_IP}"

    sh "rsync -avzh --progress #{host}:~/CarND-BehavioralCloning-P3/model*.json ."
    sh "rsync -avzh --progress #{host}:~/CarND-BehavioralCloning-P3/model*.h5 ."
  end

  task :down, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~/CarND-BehavioralCloning-P3')
    host = "carnd@#{CARND_IP}"
    puts "downloading #{host}:#{args[:src]} to #{args[:dest]}"
    sh "scp -rp #{host}:#{args[:src]} #{args[:dest]}"
  end
end
